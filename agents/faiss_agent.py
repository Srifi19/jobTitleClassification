from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import pandas as pd
import os
from helpers.data_loader import ALL_JOB_TITLES

from helpers.logger import LOGGER, LoggingLevels
from config.config import DATASET_PATH

FAISS_STORE_PATH = "cache/faiss_store"


class _FaissAgent:
    def __init__(self) -> None:
        LOGGER.log("Initializing faiss agent...", level=LoggingLevels.INFO)
        # init openAi embeddings
        self._embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        # if faiss store does not exist, load faiss agent for the first time
        if not os.path.exists(FAISS_STORE_PATH):
            self._first_time_init()

        # load the faiss store
        self._faiss = FAISS.load_local(FAISS_STORE_PATH, embeddings=self._embeddings)

    def _first_time_init(self):
        LOGGER.log(
            "Initializing faiss agent for the first time, this may take some time...",
            level=LoggingLevels.INFO,
        )
        # load the dataset
        dataset = pd.read_excel(DATASET_PATH)
        # extract needed words from dataset
        words = [v[0].strip() for v in list(dataset.values) if type(v[0]) is str]

        # words override
        words = ALL_JOB_TITLES

        # create the faiss store
        self._faiss = FAISS.from_texts(words, embedding=self._embeddings)
        # save the faiss store
        self._faiss.save_local(FAISS_STORE_PATH)

        # done
        LOGGER.log(
            "Faiss Agent is initialized for the first time, dataset is embedded and cached successfully",
            level=LoggingLevels.INFO,
        )

    def get_similar_titles(self, title: str, top_n: int = 5) -> list[tuple[str, float]]:
        title_response = [
            (doc.page_content, score)
            for doc, score in self._faiss.similarity_search_with_score(
                query=title,
                k=top_n,
            )
        ]

        return title_response[:top_n] if len(title_response) > top_n else title_response


FAISS_AGENT = _FaissAgent()
