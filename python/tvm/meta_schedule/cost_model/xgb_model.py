# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
XGBoost-based cost model
"""
from itertools import chain as itertools_chain
import logging
import os
import tempfile
from typing import Any, Callable, Dict, List, NamedTuple, Optional, TYPE_CHECKING, Tuple

import numpy as np

from ...contrib.tar import tar, untar
from ..cost_model import PyCostModel
from ..feature_extractor import FeatureExtractor
from ..runner import RunnerResult
from ..search_strategy import MeasureCandidate
from ..utils import cpu_count
from .metric import max_curve

if TYPE_CHECKING:
    from ..tune_context import TuneContext
    import xgboost as xgb


logger = logging.getLogger(__name__)


def make_metric_sorter(focused_metric):
    """ Make sure the focused metric is the first one. """

    def metric_name_for_sort(name):
        if focused_metric == name:
            return "!" + name
        return name

    def sort_key(key):
        key, _ = key
        return metric_name_for_sort(key)

    return sort_key


class PackSum:
    """The pack-sum format

    Parameters
    ----------
    dmatrix : xgb.DMatrix
        A float64 array of shape [n, m],
        where `n` is the packed number of blocks,
        and `m` is the length of feature vector on each block
    ids : np.ndarray
        An int64 array of shape [n] containing nonnegative integers,
        indicating which the index of a sample that a block belongs to
    """

    dmatrix: "xgb.DMatrix"  # type: ignore # pylint: disable=invalid-name
    ids: np.ndarray

    def __init__(
        self,
        xs: List[np.ndarray],
        ys: Optional[np.ndarray],
    ):
        """Create PackSum format given a batch of samples

        Parameters
        ----------
        xs : List[np.ndarray]
            A batch of input samples
        ys : Optional[List[float]]
            A batch of labels. None means no labels available.
        """
        import xgboost as xgb  # type: ignore # pylint: disable=import-outside-toplevel

        repeats = [x.shape[0] for x in xs]
        xs = np.concatenate(xs, axis=0)
        self.ids = np.concatenate([[i] * repeat for i, repeat in enumerate(repeats)], axis=0)
        if ys is None:
            self.dmatrix = xgb.DMatrix(data=xs, label=None)
        else:
            ys = np.concatenate([[y] * repeat for y, repeat in zip(ys, repeats)], axis=0)
            self.dmatrix = xgb.DMatrix(data=xs, label=ys)
            self.dmatrix.set_weight(ys)

    def predict_with_score(self, pred: np.ndarray) -> np.ndarray:
        """Predict the labels given the block level prediction scores.

        Parameters
        ----------
        pred : np.ndarray
            The block level predictions

        Returns
        -------
        result : np.ndarray
            The predictions for each candidate.
        """
        return np.bincount(self.ids, weights=pred)

    def obj_square_error(self, ys_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Implement square error loss on pack-sum format as
        a custom objective function for xgboost.

        Parameters
        ----------
        ys_pred: np.ndarray
            The predictions

        Returns
        -------
        gradient: np.ndarray
            The gradient according to the xgboost format
        hessian: np.ndarray
            The hessian according to the xgboost format
        """
        # Making prediction
        ys_pred = self.predict_with_score(ys_pred)
        # Propagate prediction to each block
        ys_pred = ys_pred[self.ids]
        # The gradient and hessian
        ys = self.dmatrix.get_label()  # type: ignore # pylint: disable=invalid-name
        gradient = ys_pred - ys
        hessian = np.ones_like(gradient)
        return gradient * ys, hessian * ys

    def rmse(self, ys_pred: np.ndarray) -> Tuple[str, float]:
        """Evaluate RMSE (rooted mean square error) in the pack-sum format

        Parameters
        ----------
        ys_pred: np.ndarray
            The raw predictions

        Returns
        -------
        name: str
            The name of the metric
        score: float
            The score of the metric
        """
        # Making prediction
        ys_pred = self.predict_with_score(ys_pred)
        # Propagate prediction to each block
        ys_pred = ys_pred[self.ids]
        # The RMSE
        ys = self.dmatrix.get_label()  # type: ignore # pylint: disable=invalid-name
        square_error = np.square(ys_pred - ys)
        rmse = np.sqrt(square_error.mean())
        return "p-rmse", rmse

    def average_peak_score(
        self,
        ys_pred: np.ndarray,
        n: int,
    ) -> Tuple[str, float]:
        """Evaluate average-peak-score@N in the pack-sum format

        Parameters
        ----------
        ys_pred: np.ndarray
            The raw prediction
        n : int
            The N in average-peak-score@N

        Returns
        -------
        name: str
            The name of the metric
        score: float
            The score of the metric
        """
        ys = self.dmatrix.get_label()  # type: ignore # pylint: disable=invalid-name
        ys = self.predict_with_score(ys)  # type: ignore # pylint: disable=invalid-name
        ys = ys / np.unique(self.ids, return_counts=True)[1]  # type: ignore # pylint: disable=invalid-name
        ys_pred = self.predict_with_score(ys_pred)
        trials = np.argsort(ys_pred)[::-1][:n]
        trial_scores = ys[trials]
        curve = max_curve(trial_scores) / np.max(ys)
        score = np.mean(curve)
        return f"a-peak@{n}", score


class XGBConfig(NamedTuple):
    """XGBoost model configuration

    Parameters
    ----------
    max_depth : int
        The maximum depth.
    gamma : float
        The gamma.
    min_child_weight : float
        The minimum child weight.
    eta : float
        The eta, learning rate.
    seed : int
        The random seed.
    nthread : Optional[int],
        The number of threads to use.
        Default is None, which means to use physical number of cores.
    """

    def to_dict(self):
        xgb_params = {
            "max_depth": self.max_depth,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "eta": self.eta,
            "seed": self.seed,
            "nthread": self.nthread,
        }
        return xgb_params

    max_depth: int = 10
    gamma: float = 0.001
    min_child_weight: float = 0
    eta: float = 0.2
    seed: int = 43
    nthread: Optional[int] = None


class XGBModel(PyCostModel):
    """XGBoost model

    Parameters
    ----------
    extractor : FeatureExtractor
        The feature extractor for the model.
    config : XGBConfig
        The XGBoost model config.
    num_warmup_samples : int
        The number of samples that are used for warmup, i.e., the first few samples are predicted
        with random results.
    early_stopping_rounds : int
        The number of rounds for early stopping.
    verbose_eval : int
        The verbose level when doing evaluation.
    average_peak_n : int
        The number to calculate average peak score.
    """

    # feature extractor
    extractor: FeatureExtractor
    # xgboost model config
    config: XGBConfig
    # behavior of randomness
    num_warmup_samples: int
    # evaluation
    early_stopping_rounds: int
    verbose_eval: int
    average_peak_n: int
    # states
    cached_features: List[np.ndarray]
    cached_mean_costs: np.ndarray
    cached_normalizer: Optional[float]
    booster: Optional["xgb.Booster"]

    def __init__(
        self,
        *,
        # feature extractor
        extractor: FeatureExtractor,
        # xgboost model config
        config: XGBConfig = XGBConfig(),
        # behavior of randomness
        num_warmup_samples: int = 100,
        # evaluation
        early_stopping_rounds: int = 50,
        verbose_eval: int = 25,
        average_peak_n: int = 32,
    ):
        super().__init__()
        # feature extractor
        self.extractor = extractor
        # model-related
        if config.nthread is None:
            # use physical core number
            config = config._replace(nthread=cpu_count(logical=False))
        self.config = config
        # behavior of randomness
        self.num_warmup_samples = num_warmup_samples
        # evaluation
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.average_peak_n = average_peak_n
        # states
        self.cached_features = []
        self.cached_mean_costs = np.empty((0,), dtype="float64")
        self.cached_normalizer = None
        self.booster = None

    def load(self, path: str) -> None:
        """Load the cost model from given file location.

        Parameters
        ----------
        path : str
            The file path.

        Note
        ----
        Since XGBoost model trains from scratch, each time we can only load the model without the
        previous cached features / results so any call of update won't use previous training data.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            untar(path, tmp_dir)
            self.booster.load_model(os.path.join(tmp_dir, "model.bin"))
            self.cached_features = list(
                np.load(os.path.join(tmp_dir, "cached_features.npy"), allow_pickle=True)
            )
            self.cached_mean_costs = np.load(
                os.path.join(tmp_dir, "cached_mean_costs.npy"), allow_pickle=True
            )
            self._set_cached_normalizer()

    def save(self, path: str) -> None:
        """Save the cost model to given file location.

        Parameters
        ----------
        path : str
            The file path.

        Note
        ----
        Since XGBoost model trains from scratch, each time we can only save the model without the
        previous cached features / results so any call of update won't use previous training data.
        """
        import xgboost as xgb  # pylint: disable=import-outside-toplevel

        if self.booster is None:
            # save all the paramaters
            self.booster = xgb.Booster(self.config.to_dict())
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.booster.save_model(os.path.join(tmpdirname, "model.bin"))
            np.save(
                os.path.join(tmpdirname, "cached_features.npy"),
                np.array(self.cached_features, dtype=object),
            )
            np.save(os.path.join(tmpdirname, "cached_mean_costs.npy"), self.cached_mean_costs)
            tar(
                path,
                [
                    os.path.join(tmpdirname, "model.bin"),
                    os.path.join(tmpdirname, "cached_features.npy"),
                    os.path.join(tmpdirname, "cached_mean_costs.npy"),
                ],
            )

    def update(
        self,
        tune_context: "TuneContext",
        candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        """Update the cost model given running results.

        Parameters
        ----------
        tune_context : TuneContext
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.
        results : List[RunnerResult]
            The running results of the measure candidates.
        """
        assert len(candidates) == len(results)
        if len(candidates) == 0:
            return
        # extract feature and do validation

        def _mean_cost(x: RunnerResult) -> float:
            if not x.run_secs:
                return 1e10
            return float(np.median([float(s) for s in x.run_secs]))

        new_features = [
            x.numpy().astype("float32")
            for x in self.extractor.extract_from(tune_context, candidates)
        ]
        new_mean_costs = np.asarray(
            [_mean_cost(x) for x in results],
            dtype="float32",
        )
        if self.booster is not None and self.cached_normalizer is not None:
            logger.debug(
                "XGB validation: %s",
                "\t".join(
                    f"{key}: {score:.6f}"
                    for key, score in self._validate(
                        xs=new_features,
                        ys=new_mean_costs,
                    )
                ),
            )
        # use together with previous features
        self.cached_features.extend(new_features)
        self.cached_mean_costs = np.append(self.cached_mean_costs, new_mean_costs)
        self._set_cached_normalizer()
        # train xgb model
        self._train(
            xs=self.cached_features,
            ys=self.cached_mean_costs,
        )

    def predict(
        self, tune_context: "TuneContext", candidates: List[MeasureCandidate]
    ) -> np.ndarray:
        """Predict the normalized score using the cost model.

        Parameters
        ----------
        tune_context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.

        Return
        ------
        result : np.ndarray
            The predicted normalized score.
        """
        n_measured = len(self.cached_features)
        if self.booster is not None and n_measured >= self.num_warmup_samples:
            features = self.extractor.extract_from(tune_context, candidates)
            ret = self._predict(xs=[x.numpy().astype("float32") for x in features])
        else:
            ret = np.random.uniform(
                low=0,
                high=1,
                size=(len(candidates),),
            )
        return ret.astype("float64")

    def _train(  # type: ignore # pylint: disable=invalid-name
        self,
        xs: List[np.ndarray],
        ys: np.ndarray,
    ) -> None:
        import xgboost as xgb  # type: ignore # pylint: disable=import-outside-toplevel

        self.d_train = PackSum(
            xs=xs,
            ys=self.cached_normalizer / ys,
        )

        def obj(ys_pred: np.ndarray, d_train: "xgb.DMatrix"):  # type: ignore # pylint: disable = unused-argument
            return self.d_train.obj_square_error(ys_pred)

        def rmse(ys_pred: np.ndarray, d_train: "xgb.DMatrix"):  # type: ignore # pylint: disable = unused-argument
            return self.d_train.rmse(ys_pred)

        def average_peak_score(
            ys_pred: np.ndarray, d_train: "xgb.DMatrix"  # type: ignore # pylint: disable = unused-argument
        ):
            return self.d_train.average_peak_score(ys_pred, self.average_peak_n)

        self.booster = xgb.train(
            self.config.to_dict(),
            self.d_train.dmatrix,
            num_boost_round=10000,
            obj=obj,
            callbacks=[
                custom_callback(
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=self.verbose_eval,
                    fevals=[
                        rmse,
                        average_peak_score,
                    ],
                    evals=[(self.d_train.dmatrix, "tr")],
                )
            ],
        )

        del self.d_train
        # todo(zxybazh): measure callback to save the model

    def _predict(  # type: ignore # pylint: disable=invalid-name
        self,
        xs: List[np.ndarray],
    ) -> np.ndarray:
        d_test = PackSum(xs=xs, ys=None)
        pred = self.booster.predict(d_test.dmatrix)
        ret = d_test.predict_with_score(pred)
        return ret

    def _validate(  # type: ignore # pylint: disable=invalid-name
        self,
        xs: List[np.ndarray],
        ys: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """Evaluate the score of inputs.

        Parameters
        ----------
        xs : List[np.ndarray]
            A batch of input samples
        ys : List[float]
            A batch of labels

        Returns
        -------
        scores: np.ndarray
            The predicted result for all inputs.
        """
        if self.booster is None or self.cached_normalizer is None:
            return []

        d_valid = PackSum(
            xs=xs,
            ys=self.cached_normalizer / ys,
        )

        def average_peak_score(ys_pred: np.ndarray):
            return d_valid.average_peak_score(ys_pred, n=self.average_peak_n)

        ys_pred = self.booster.predict(d_valid.dmatrix)
        eval_result: List[Tuple[str, float]] = [
            feval(ys_pred)
            for feval in (
                average_peak_score,
                d_valid.rmse,
            )
        ]
        eval_result.sort(key=make_metric_sorter("p-rmse"))
        return eval_result

    def _set_cached_normalizer(self) -> None:
        filtered = self.cached_mean_costs[self.cached_mean_costs > 0]
        if filtered.size == 0:
            self.cached_normalizer = 1.0
        else:
            self.cached_normalizer = np.min(filtered)
            assert self.cached_normalizer > 0


def custom_callback(
    early_stopping_rounds: int,
    verbose_eval: int,
    fevals: List[Callable],
    evals: List[Tuple["xgb.DMatrix", str]],
    focused_metric: str = "tr-p-rmse",
):
    """Callback function for xgboost to support multiple custom evaluation functions"""
    sort_key = make_metric_sorter(focused_metric=focused_metric)

    state: Dict[str, Any] = {}

    def init(env: "xgb.core.CallbackEnv"):
        """Internal function"""
        booster: "xgb.Booster" = env.model

        state["best_iteration"] = 0
        state["best_score"] = float("inf")
        if booster is None:
            assert env.cvfolds is not None
            return
        if booster.attr("best_score") is not None:
            state["best_score"] = float(booster.attr("best_score"))
            state["best_iteration"] = int(booster.attr("best_iteration"))
            state["best_msg"] = booster.attr("best_msg")
        else:
            booster.set_attr(best_iteration=str(state["best_iteration"]))
            booster.set_attr(best_score=str(state["best_score"]))

    def callback(env: "xgb.core.CallbackEnv"):
        # pylint:disable = import-outside-toplevel
        import xgboost as xgb
        from xgboost.callback import _fmt_metric
        from xgboost.core import EarlyStopException

        try:
            from xgboost.training import aggcv
        except ImportError:
            from xgboost.callback import _aggcv as aggcv
        # pylint:enable = import-outside-toplevel

        if not state:
            init(env)
        booster: xgb.Booster = env.model
        iteration: int = env.iteration
        cvfolds: List[xgb.training.CVPack] = env.cvfolds
        ##### Evaluation #####
        # `eval_result` is a list of (key, score)
        eval_result: List[Tuple[str, float]] = []
        if cvfolds is None:
            eval_result = list(
                itertools_chain.from_iterable(
                    [
                        (key, float(value))
                        for key, value in map(
                            lambda x: x.split(":"),
                            booster.eval_set(
                                evals=evals,
                                iteration=iteration,
                                feval=feval,
                            ).split()[1:],
                        )
                    ]
                    for feval in fevals
                )
            )
        else:
            eval_result = list(
                itertools_chain.from_iterable(
                    [
                        (key, score)
                        for key, score, _std in aggcv(
                            fold.eval(
                                iteration=iteration,
                                feval=feval,
                            )
                            for fold in cvfolds
                        )
                    ]
                    for feval in fevals
                )
            )
        eval_result = list(eval_result)
        eval_result.sort(key=sort_key)

        ##### Print eval result #####
        if verbose_eval and iteration % verbose_eval == 0:
            info = []
            for key, score in eval_result:
                if "null" not in key:
                    info.append(f"{key}: {score:.6f}")
            logger.debug("XGB iter %3d: %s", iteration, "\t".join(info))

        ##### Choose score and do early stopping #####
        score = None
        for key, _score in eval_result:
            if key == focused_metric:
                score = _score
                break
        assert score is not None

        best_score = state["best_score"]
        best_iteration = state["best_iteration"]
        if score < best_score:
            tab = "\t"  # to work with f-string
            msg = f"[{env.iteration}] {tab.join([_fmt_metric(x) for x in eval_result])}"
            state["best_msg"] = msg
            state["best_score"] = score
            state["best_iteration"] = env.iteration
            # save the property to attributes, so they will occur in checkpoint.
            if env.model is not None:
                env.model.set_attr(
                    best_score=str(state["best_score"]),
                    best_iteration=str(state["best_iteration"]),
                    best_msg=state["best_msg"],
                )
        elif env.iteration - best_iteration >= early_stopping_rounds:
            best_msg = state["best_msg"]
            if verbose_eval and env.rank == 0:
                logger.debug("XGB stopped. Best iteration: %s ", best_msg)
            raise EarlyStopException(best_iteration)

    return callback
