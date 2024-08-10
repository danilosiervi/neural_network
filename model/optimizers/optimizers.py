from model.optimizers.sgd import OptimizerSGD
from model.optimizers.ada_grad import OptimizerAdaGrad
from model.optimizers.rms_prop import OptimizerRMSProp
from model.optimizers.adam import OptimizerAdam


ALL_OPTIMIZERS = {
    OptimizerSGD,
    OptimizerAdaGrad,
    OptimizerRMSProp,
    OptimizerAdam,
}

ALL_OPTIMIZERS_DICT = {fn.__name__: fn for fn in ALL_OPTIMIZERS}


def get_optimizer(identifier):
    if identifier is None:
        return OptimizerAdam

    if isinstance(identifier, str):
        obj = ALL_ACTIVATIONS_DICT.get(identifier, None)
    else:
        obj = identifier

    if callable(obj):
        return obj
