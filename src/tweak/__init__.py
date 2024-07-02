import logging

from .predict import *
from .utils import *

from tunip.logger import init_logging_handler
from tunip.orjson_utils import *


LOGGER = init_logging_handler(name="tweak", level=logging.DEBUG)

DEFAULT_PADDING = "longest"