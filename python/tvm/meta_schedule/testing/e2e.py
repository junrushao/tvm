import multiprocessing
import os
import pickle
from typing import Dict, List, Optional, Tuple

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.ir import IRModule
from tvm.runtime import NDArray, load_param_dict, save_param_dict

SUPPORTED = [
    # TorchVision
    "resnet_18",
    "resnet_50",
    "mobilenet_v2",
    "mobilenet_v3",
    "wide_resnet_50",
    "resnext_50",
    "resnet3d_18",
    "inception_v3",
    "densenet_121",
    "vgg_16",
    # Transformer
    "bert_tiny",
    "bert_base",
    "bert_medium",
    "bert_large",
    # Relay testing
    "dcgan",
]


def _get_network(
    args: Tuple[str, List[int]]
) -> Tuple[IRModule, bytearray, Tuple[str, List[int], str]]:
    name: str
    input_shape: List[int]
    name, input_shape = args

    mod: IRModule

    if name in [
        "resnet_18",
        "resnet_50",
        "wide_resnet_50",
        "resnext_50",
        "mobilenet_v2",
        "mobilenet_v3",
        "inception_v3",
        "densenet_121",
        "resnet3d_18",
        "vgg_16",
    ]:
        # torchvision>=0.9.0
        import torch  # type: ignore
        import torchvision.models as models  # type: ignore

        if name in ["resnet_18", "resnet_50"]:
            model = getattr(models, name.replace("_", ""))(pretrained=False)
        elif name == "wide_resnet_50":
            model = getattr(models, "wide_resnet50_2")(pretrained=False)
        elif name == "resnext_50":
            model = getattr(models, "resnext50_32x4d")(pretrained=False)
        elif name == "mobilenet_v2":
            model = getattr(models, name)(pretrained=False)
        elif name == "mobilenet_v3":
            model = getattr(models, name + "_large")(pretrained=False)
        elif name == "inception_v3":
            model = getattr(models, name)(pretrained=False, aux_logits=False)
        elif name == "densenet_121":
            model = getattr(models, name.replace("_", ""))(pretrained=False)
        elif name == "resnet3d_18":
            model = models.video.r3d_18(pretrained=False)
        elif name == "vgg_16":
            model = getattr(models, name.replace("_", ""))(pretrained=False)

        dtype = "float32"
        input_data = torch.randn(input_shape).type(
            {
                "float32": torch.float32,
            }[dtype]
        )
        scripted_model = torch.jit.trace(model, input_data).eval()
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        with tvm.transform.PassContext(opt_level=3):
            mod = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(
                        {
                            "nn.conv2d": ["NHWC", "default"],
                            "nn.conv3d": ["NDHWC", "default"],
                            "nn.max_pool2d": ["NHWC", "default"],
                            "nn.avg_pool2d": ["NHWC", "default"],
                        }
                    ),
                ]
            )(mod)
        inputs = (input_name, input_shape, dtype)
    elif name in ["bert_tiny", "bert_base", "bert_medium", "bert_large"]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # pip3 install transformers==3.5 torch==1.7
        import torch  # type: ignore
        import transformers  # type: ignore

        config_dict = {
            "bert_tiny": transformers.BertConfig(
                num_hidden_layers=6,
                hidden_size=512,
                intermediate_size=2048,
                num_attention_heads=8,
            ),
            "bert_base": transformers.BertConfig(
                num_hidden_layers=12,
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
            ),
            "bert_medium": transformers.BertConfig(
                num_hidden_layers=12,
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
            ),
            "bert_large": transformers.BertConfig(
                num_hidden_layers=24,
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
            ),
        }
        configuration = config_dict[name]
        model = transformers.BertModel(configuration)
        input_name = "input_ids"
        input_dtype = "int64"
        A = torch.randint(10000, input_shape)
        model.eval()
        scripted_model = torch.jit.trace(model, [A], strict=False)
        input_name = "input_ids"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        mod = relay.transform.FastMath()(mod)
        mod = relay.transform.CombineParallelBatchMatmul()(mod)
        inputs = (input_name, input_shape, input_dtype)
    elif name == "dcgan":
        output_shape = input_shape
        batch_size = output_shape[0]
        oshape = output_shape[1:]
        mod, params = relay.testing.dcgan.get_workload(
            batch_size=batch_size,
            oshape=oshape,
            layout="NHWC",
        )
        inputs = ("data", [100], "float32")
    else:
        raise ValueError("Invalid name: " + name)

    params_bytearray: bytearray = save_param_dict(params)
    return mod, params_bytearray, inputs


def get_network(
    name: str,
    input_shape: List[int],
    cache_dir: Optional[str] = None,
) -> Tuple[IRModule, Dict[str, NDArray], Tuple[str, List[int], str]]:
    mod: IRModule
    params_bytearray: bytearray
    params: Dict[str, NDArray]
    inputs: Tuple[str, List[int], str]

    keyword = f"{name}-{input_shape}.json"
    if cache_dir is not None:
        path = os.path.join(cache_dir, keyword)
        if os.path.exists(path):
            print(f"Load cached network file: {path}")
            with open(path, "rb") as i_f:
                mod, params_bytearray, inputs = pickle.load(i_f)
            params = load_param_dict(params_bytearray)
            return mod, params, inputs
    with multiprocessing.Pool(processes=1) as pool:
        result = pool.map(_get_network, [(name, input_shape)])
        ((mod, params_bytearray, inputs),) = result
        params = load_param_dict(params_bytearray)
    if cache_dir is not None:
        path = os.path.join(cache_dir, keyword)
        with open(path, "wb") as o_f:
            pickle.dump((mod, params_bytearray, inputs), o_f)
    return mod, params, inputs
