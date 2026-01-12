from model.activations.linear import Linear
from model.activations.relu import Relu
from model.activations.softmax import Softmax
from model.activations.sigmoid import Sigmoid
from model.activations.tanh import Tanh


ALL_ACTIVATIONS = {
    Linear,
    Relu,
    Tanh,
    Softmax,
    Sigmoid,
}

ALL_ACTIVATIONS_DICT = {fn.__name__: fn for fn in ALL_ACTIVATIONS}


def get_activation(identifier):
    if identifier is None:
        return Linear

    if isinstance(identifier, str):
        obj = ALL_ACTIVATIONS_DICT.get(identifier, None)
    else:
        obj = identifier

    if callable(obj):
        return obj
    return None
