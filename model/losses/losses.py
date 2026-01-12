from model.losses.mean_squared_error import MeanSquaredError
from model.losses.mean_absolute_error import MeanAbsoluteError
from model.losses.categorical_crossentropy import CategoricalCrossentropy
from model.losses.binary_crossentropy import BinaryCrossentropy


ALL_LOSSES = {
    MeanSquaredError,
    MeanAbsoluteError,
    CategoricalCrossentropy,
    BinaryCrossentropy,
}

ALL_LOSSES_DICT = {fn.__name__: fn for fn in ALL_LOSSES}


def get_loss(identifier):
    if identifier is None:
        return MeanSquaredError

    if isinstance(identifier, str):
        obj = ALL_LOSSES_DICT.get(identifier, None)
    else:
        obj = identifier

    if callable(obj):
        return obj
    return None
