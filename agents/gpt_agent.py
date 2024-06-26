#! this agent use FAISS agent, so it should be imported after FAISS agent to avoid
# circular imports error

from dataclasses import dataclass
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

from .faiss_agent import FAISS_AGENT

from helpers.logger import LOGGER, LoggingLevels
from tools.courses_scraper import CoursesScraper, ScrapedCourse
from config.prompting_config import *


@dataclass
class GptAgentResponse:
    response: any
    input_tokens: int
    output_tokens: int
    cost: float


class _GptAgent:
    def __init__(self):
        LOGGER.log("Initializing GPT agent...", level=LoggingLevels.INFO)
        # chat model
        self._chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        # courses scraper for course suggestion method
        self._courses_scraper = CoursesScraper(headless=False)
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

    def generate_recommended_paths(
        self,
        hard_skills: list[str],
        soft_skills: list[str],
        tools: list[str],
        education: list[str],
        experience: list[str],
        ignore_titles: list[str],
    ) -> GptAgentResponse:
        LOGGER.log(
            f"Generating recommended paths using GPT agent...\n't - Hard Skills: {hard_skills}\n\t - Soft Skills: {soft_skills}",
            level=LoggingLevels.INFO,
        )
        # format the input
        model_input = PATHS_GENERATION_PROMPT.format_prompt(
            hard_skills=hard_skills,
            soft_skills=soft_skills,
            education=education,
            experience=experience,
            tools=tools,
            ignore_titles=ignore_titles,
        )
        # OPENAI Callback to track the cost
        with get_openai_callback() as callback:
            # run the chain
            response = PATHS_GENERATION_OUTPUT_PARSER.parse(
                self._chat_model.call_as_llm(model_input.to_string())
            )
            LOGGER.log(
                f"User: {model_input.to_string()}\nAI: {response}",
                level=LoggingLevels.INFO,
            )
            return GptAgentResponse(
                response=response,
                input_tokens=callback.prompt_tokens,
                output_tokens=callback.completion_tokens,
                cost=callback.total_cost,
            )

    def generate_path_requirements(
        self,
        path: str,
        hard_skills: list[str],
        soft_skills: list[str],
        tools: list[str],
        education: list[str],
        experience: list[str],
    ) -> GptAgentResponse:
        LOGGER.log(
            f"Generating path requirements using GPT agent...\n't - Hard Skills: {hard_skills}\n\t - Soft Skills: {soft_skills}",
            level=LoggingLevels.INFO,
        )
        # format the input
        model_input = PATHS_REQUIREMENTS_GENERATION_PROMPT.format_prompt(
            path=path,
            hard_skills=hard_skills,
            soft_skills=soft_skills,
            education=education,
            tools=tools,
            experience=experience,
        )
        # OPENAI Callback to track the cost
        with get_openai_callback() as callback:
            # run the chain
            response = PATHS_REQUIREMENTS_GENERATION_OUTPUT_PARSER.parse(
                self._chat_model.call_as_llm(model_input.to_string())
            )
            LOGGER.log(
                f"User: {model_input.to_string()}\nAI: {response}",
                level=LoggingLevels.INFO,
            )
            return GptAgentResponse(
                response=response,
                input_tokens=callback.prompt_tokens,
                output_tokens=callback.completion_tokens,
                cost=callback.total_cost,
            )

    def recommend_courses(
        self,
        topic: str,
        hard_skills: list[str],
        tools: list[str],
    ):
        # get top available courses from ClassCentral
        available_courses = self._courses_scraper.get_top_courses(topic=topic)
        # generate json-like object for available course
        available_courses_dict = [
            {"id": i, "title": course.title, "description": course.description}
            for i, course in enumerate(available_courses)
        ]

        # format the input
        model_input = SUGGESTED_COURSE_PROMPT.format_prompt(
            courses=available_courses_dict,
            career_goal=topic,
            hard_skills=hard_skills,
            tools=tools,
        )

        # OPENAI Callback to track the cost
        with get_openai_callback() as callback:
            # run the chain
            response = SUGGESTED_COURSE_OUTPUT_PARSER.parse(
                self._chat_model.call_as_llm(model_input.to_string())
            ).suggested_courses
            print(f"{response = }")
            response_list = [
                {
                    "title": available_courses[r.course_id].title,
                    "description": available_courses[r.course_id].description,
                    "link": available_courses[r.course_id].url,
                    "reason": r.suggestion_reason,
                }
                for r in response
            ]
            LOGGER.log(
                f"User: {model_input.to_string()}\nAI: {response}",
                level=LoggingLevels.INFO,
            )
            return GptAgentResponse(
                response=response_list,
                input_tokens=callback.prompt_tokens,
                output_tokens=callback.completion_tokens,
                cost=callback.total_cost,
            )


GPT_AGENT = _GptAgent()
