import logging

logger = logging.getLogger(__package__)

if logger.hasHandlers():
    logger.handlers.clear()

handler = logging.StreamHandler()

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s]: %(message)s",
    "%H:%M:%S %z",
)

handler.setFormatter(formatter)

logger.addHandler(handler)

logger.setLevel(logging.INFO)
