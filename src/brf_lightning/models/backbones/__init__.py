# from .simple_brf_v2 import BRFStackedEncoderWithSpeed
# from .stacked_brf import BRFStackedEncoder
# from .mvp import MinimalBRFEncoder
from .vanilla_brf import SimpleResRNN
__all__ = [
    # "BRFStackedEncoderWithSpeed",
    # "BRFStackedEncoder",
    # "MinimalBRFEncoder",
    "SimpleResRNN"
]