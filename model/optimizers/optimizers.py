from model.optimizers.sgd import SGD
from model.optimizers.ada_grad import AdaGrad
from model.optimizers.rms_prop import RMSProp
from model.optimizers.adam import Adam


ALL_OPTIMIZERS = {
    SGD,
    AdaGrad,
    RMSProp,
    Adam,
}

ALL_OPTIMIZERS_DICT = {fn.__name__: fn for fn in ALL_OPTIMIZERS}


def get_optimizer(identifier):
    if identifier is None:
        return Adam

    if isinstance(identifier, str):
        obj = ALL_OPTIMIZERS_DICT.get(identifier, None)
    else:
        obj = identifier

    if callable(obj):
        return obj
    return None
