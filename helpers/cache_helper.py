from pickle import load, dump

from config.config import CACHE_PATH
from helpers.logger import LOGGER, LoggingLevels


CACHE_FILE_LOCATION = rf"{CACHE_PATH}/cache.pkl"


class CacheHelper:
    @staticmethod
    def _init_cache_file():
        """raise an Exception in case of failure"""
        with open(CACHE_FILE_LOCATION, "w+") as f:
            dump({}, f)

    @staticmethod
    def _get_cache_map() -> dict:
        """raise Exception in case of any Exception different than the `FileNotFoundError`"""
        LOGGER.log("Loading cache map", level=LoggingLevels.INFO)
        try:
            with open(CACHE_FILE_LOCATION, "rb") as f:
                return load(f)
        except FileNotFoundError:
            LOGGER.log(
                "Cache file is not found, initializing new one...",
                level=LoggingLevels.INFO,
            )
            # init cache file and map
            CacheHelper._init_cache_file()
            return {}
        except Exception as e:
            LOGGER.log(f"Failed to load cache map: {e}", level=LoggingLevels.ERROR)
            return {}

    @staticmethod
    def load_from_cache(key: str) -> any:
        LOGGER.log(
            f"Loading data with key '{key}' from cache", level=LoggingLevels.INFO
        )
        try:
            with open(CACHE_FILE_LOCATION, "rb") as f:
                return load(f)[key]
        except Exception as e:
            LOGGER.log(
                f"Failed to load data with key '{key}' from cache: {e}",
                level=LoggingLevels.ERROR,
            )
            return None

    @staticmethod
    def save_to_cache(key: str, value: any) -> bool:
        LOGGER.log(f"Caching new data for the key '{key}'", level=LoggingLevels.INFO)
        try:
            cache_map = CacheHelper._get_cache_map()
            cache_map[key] = value

            # save the updated map
            with open(CACHE_FILE_LOCATION, "wb+") as f:
                dump(cache_map, f)
                LOGGER.log(
                    f"new data is successfully cached with key '{key}'",
                    level=LoggingLevels.INFO,
                )
            return True

        except Exception as e:
            LOGGER.log(f"Failed to cache data with key '{key}': {e}")
            return False
