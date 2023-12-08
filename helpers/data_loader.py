import pandas as pd

from helpers.logger import LOGGER, LoggingLevels
from config.config import DATASET_PATH

LOGGER.log("Loading saved Job titles...", level=LoggingLevels.INFO)

# ALL_JOB_TITLES = [
#     v[0].strip() for v in list(pd.read_excel(DATASET_PATH).values) if type(v[0]) is str
# ]
ALL_JOB_TITLES = [
    "Agriculture",
    "Animal science",
    "Business",
    "Cosmetology",
    "Customer service",
    "Creative",
    "Education",
    "Engineering",
    "Information technology",
    "Finance",
    "Health care",
    "Hospitality",
    "Human resources",
    "Leadership",
    "Marketing",
    "Operations",
    "Sales",
]
