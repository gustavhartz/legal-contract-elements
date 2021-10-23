import logging
import sys
import uuid

from pythonjsonlogger import jsonlogger

format_str = '%(message)%(levelname)%(name)%(asctime)'
FORMATTER = jsonlogger.JsonFormatter(format_str)
LOG_FILE = "my_app.log"


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)  # better to have too much log than not enough
    logger.addHandler(get_console_handler())
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False

    return logger


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("test")
    logger.error("Nice example to show excellent work in JIRA :D ", extra={'RequestUUDI': uuid.uuid4()})
