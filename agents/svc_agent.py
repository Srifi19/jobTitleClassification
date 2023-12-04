from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

from helpers.logger import LOGGER, LoggingLevels
from helpers.cache_helper import CacheHelper

DATASET_PATH = "data/svc_dataset.csv"


class _SvcAgent:
    def __init__(self) -> None:
        LOGGER.log("Initializing SVC agent...", level=LoggingLevels.INFO)
        # load the TF-IDF vectorizer
        self._vectorizer = CacheHelper.load_from_cache("tfidf_vectorizer")

        # load the model
        LOGGER.log("Loading cached SVC model...", level=LoggingLevels.INFO)
        self._model = CacheHelper.load_from_cache("svc_model")

        # if the model is not cached, load it for the first time
        if self._model is None or self._vectorizer is None:
            LOGGER.log(f"No cached SVC mdoel is found!", level=LoggingLevels.WARNING)
            self._first_time_init()

    def _first_time_init(self):
        LOGGER.log(
            "Initializing SVC agent for the first time, this may take some time...",
            level=LoggingLevels.INFO,
        )

        # init the vectorizer
        self._vectorizer = TfidfVectorizer()

        # load the dataset
        # ** the dataset should be a .csv file with two columns: example, label **#
        data: pd.DataFrame = pd.read_csv(DATASET_PATH)

        # separate features and labels
        job_titles = data["example"].tolist()
        labels = data["label"].tolist()

        # split the data (train and test sets)
        X_train, X_test, y_train, y_test = train_test_split(
            job_titles, labels, test_size=0.2, random_state=42
        )

        # vectorize the data
        X_train_vectorized = self._vectorizer.fit_transform(X_train)
        X_test_vectorized = self._vectorizer.transform(X_test)

        # train the model
        self._model = SVC(kernel="linear", probability=True)
        self._model.fit(X_train_vectorized, y_train)

        # evaluate the model
        y_pred = self._model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)

        # log the results
        LOGGER.log(
            f"SVC Model is trained successfully... accuracy = {accuracy*100:.2f}%",
            level=LoggingLevels.INFO,
        )

        # log the classification report
        LOGGER.log(
            f"Classification Report:\n{classification_report(y_test, y_pred)}",
            level=LoggingLevels.INFO,
        )

        # cache the model
        LOGGER.log("Caching the SVC model...", level=LoggingLevels.INFO)
        try:
            CacheHelper.save_to_cache("svc_model", self._model)
            CacheHelper.save_to_cache("tfidf_vectorizer", self._vectorizer)
        except Exception as e:
            LOGGER.log(
                f"Failed to cache the SVC model: {e}!", level=LoggingLevels.ERROR
            )
            raise e

        # done
        LOGGER.log(
            "SVC Agent is initialized for the first time, dataset is embedded and cached successfully",
            level=LoggingLevels.INFO,
        )

    def predict_job_title(self, title: str, top_n: int) -> str:
        LOGGER.log(f"Predicting job title for '{title}'", level=LoggingLevels.INFO)
        # convert the text to TF-IDF
        title_vectorized = self._vectorizer.transform([title])

        # predict the label
        predicted_labels = self._model.predict(title_vectorized)

        return (
            predicted_labels[:top_n]
            if len(predicted_labels) > top_n
            else predicted_labels
        )


SVC_AGENT = _SvcAgent()
