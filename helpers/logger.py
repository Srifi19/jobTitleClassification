from dataclasses import dataclass
import json
import logging
from enum import Enum

# path to config file
CONFIG_PATH = "config/logging_config.json"


class LoggingLevels(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LoggerConfig:
    log_file: str
    debug_mode: bool


def get_logger_config() -> LoggerConfig:
    """get logger config from config file"""
    with open(CONFIG_PATH, "r+") as config_file:
        json_dict = json.load(config_file)
        return LoggerConfig(
            log_file=json_dict["log_file"],
            debug_mode=json_dict["debug_mode"],
        )


class _Logger:
    def __init__(self, config: LoggerConfig):
        self._config = config
        # get logger config
        try:
            logging.basicConfig(
                level=logging.DEBUG if config.debug_mode else logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(config.log_file),
                    logging.StreamHandler(),
                ],
            )
            self.TAB_SPACES = "   "

        # if the config file is not found
        except FileNotFoundError:
            raise FileNotFoundError(f"Unable to find '{CONFIG_PATH}'")

        # if unexpected error occurred
        except Exception as e:
            raise Exception(f"Failed to initialize logger: {e}")

    def log(
        self,
        message: str,
        indentation: int = 0,
        level: LoggingLevels = LoggingLevels.INFO,
    ):
        message = (self.TAB_SPACES * indentation) + str(message)
        if level == LoggingLevels.DEBUG:
            return logging.debug(msg=message)
        if level == LoggingLevels.WARNING:
            return logging.warning(msg=message)
        if level == LoggingLevels.ERROR:
            return logging.error(msg=message)
        if level == LoggingLevels.CRITICAL:
            return logging.critical(msg=message)
        # default
        return logging.info(msg=message)


LOGGER = _Logger(config=get_logger_config())
