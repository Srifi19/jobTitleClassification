#! this agent use FAISS agent, so it should be imported after FAISS agent to avoid
# circular imports error

from dataclasses import dataclass
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

from .faiss_agent import FAISS_AGENT

from helpers.logger import LOGGER, LoggingLevels
from config.prompting_config import *


@dataclass
class GptAgentResponse:
    response: str
    input_tokens: int
    output_tokens: int
    cost: float


class _GptAgent:
    def __init__(self):
        LOGGER.log("Initializing GPT agent...", level=LoggingLevels.INFO)
        # chat model
        self._chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        # example prompt
        self._example_prompt = PromptTemplate(
            input_variables=["query", "answer"], template=EXAMPLE_TEMPLATE
        )
        # few shot prompt
        self._few_shot_prompt = FewShotPromptTemplate(
            examples=EXAMPLES,
            example_prompt=self._example_prompt,
            prefix=PREFIX,
            suffix=SUFFIX,
            input_variables=["query"],
            example_separator="\n\n",
        )
        # LLM chain
        self._chain = LLMChain(
            llm=self._chat_model,
            prompt=self._few_shot_prompt,
        )

    def get_best_job_title(
        self,
        job_title: str,
        top_n: int = 5,
    ) -> GptAgentResponse:
        # get similar titles using FAISS agent
        faiss_results: list[str] = [
            title for title, _ in FAISS_AGENT.get_similar_titles(job_title, top_n=top_n)
        ]

        # formulate the user query
        query = f"{job_title}, {faiss_results}"

        # OPENAI Callback to track the cost
        with get_openai_callback() as callback:
            # run the chain
            response = self._chain.run(query)
            LOGGER.log(f"User: {query}\nAI: {response}", level=LoggingLevels.INFO)

        return GptAgentResponse(
            response=response,
            input_tokens=callback.prompt_tokens,
            output_tokens=callback.completion_tokens,
            cost=callback.total_cost,
        )


GPT_AGENT = _GptAgent()