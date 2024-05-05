import logging

logger = logging.getLogger(__package__)

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        f"[%(asctime)s] [%(levelname)s]: %(message)s",
        "%H:%M:%S %z",
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

logger.setLevel(logging.INFO)
