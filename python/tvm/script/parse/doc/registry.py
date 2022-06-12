import ast
import inspect
from collections import defaultdict
from typing import Callable, Dict, Optional

from . import doc

FnToDoc = Callable[[ast.AST], doc.AST]
FnFromDoc = Callable[[doc.AST], ast.AST]


class Entry:
    to_doc: Optional[FnToDoc]
    from_doc: Optional[FnFromDoc]

    def __init__(self):
        self.to_doc = None
        self.from_doc = None


class Registry:
    _inst: Optional["Registry"] = None
    table: Dict[str, Entry]

    def __init__(self):
        self.table = defaultdict(Entry)


def register_to_doc(name: str):
    def f(to_doc: FnToDoc):  # pylint: disable=redefined-outer-name
        reg = Registry._inst  # pylint: disable=protected-access
        reg.table[name].to_doc = to_doc

    return f


def register_from_doc(name: str):
    def f(to_doc: FnFromDoc):  # pylint: disable=redefined-outer-name
        reg = Registry._inst  # pylint: disable=protected-access
        reg.table[name].from_doc = to_doc

    return f


def _is_atomic_type(node):
    return (
        node is None
        or node in [..., True, False]
        or isinstance(
            node,
            (
                int,
                float,
                str,
                bool,
                bytes,
                complex,
            ),
        )
    )


def _get_registry_entry(cls_name, attr):
    cls_name = cls_name.split(".")[-1]
    reg = Registry._inst  # pylint: disable=protected-access
    if cls_name in reg.table:
        entry = reg.table[cls_name]
        return getattr(entry, attr, None)
    return None


def from_doc(node):
    if _is_atomic_type(node):
        return node
    if isinstance(node, tuple):
        return tuple(from_doc(n) for n in node)
    if isinstance(node, list):
        return [from_doc(n) for n in node]
    func = _get_registry_entry(node.__class__.__name__, "from_doc")
    if not func:
        raise NotImplementedError(f"from_doc is not implemented for: {node.__class__.__name__}")
    return func(node)


def to_doc(node):
    if _is_atomic_type(node):
        return node
    if isinstance(node, tuple):
        return tuple(to_doc(n) for n in node)
    if isinstance(node, list):
        return [to_doc(n) for n in node]
    func = _get_registry_entry(node.__class__.__name__, "to_doc")
    if not func:
        raise NotImplementedError(f"to_doc is not implemented for: {node.__class__.__name__}")
    return func(node)


def _register_default():
    class DefaultTranslator:
        def __init__(self, name, func):
            self.doc_cls = getattr(doc, name)
            self.func = func

        def __call__(self, node):
            kv = {attr: getattr(node, attr) for attr in self.doc_cls._FIELDS}
            return self.doc_cls(**kv)

    Registry._inst = Registry()  # pylint: disable=protected-access
    for cls_name in dir(doc):
        doc_cls = getattr(doc, cls_name)
        if inspect.isclass(doc_cls) and issubclass(doc_cls, doc.AST):
            assert "." not in cls_name
            register_to_doc(cls_name)(DefaultTranslator(cls_name, to_doc))
            register_from_doc(cls_name)(DefaultTranslator(cls_name, from_doc))


_register_default()
