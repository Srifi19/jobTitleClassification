# =================================================== #
# --------------- Job Title Selection --------------- #
# =================================================== #
# examples for few shot learning
EXAMPLES = [
    {
        "query": "Python engineer, [Teacher, Software Developer, Scientist, Worker]",
        "answer": "Software Developer",
    },
    {
        "query": "Science Instructor, [Teacher, Software Developer, Scientist, Worker]",
        "answer": "Teacher",
    },
]

# template for example prompt
EXAMPLE_TEMPLATE = """
User: {query}
AI: {answer}
"""

# few shot learning prompt prefix
PREFIX = """
The following are excerpts from conversations with an AI assistant.
The assistant is helpful, and clever. From a given job title, and a list of some job titles, 
he can know which job title is the most similar to the given job title. Here are some examples:
"""

# few shot learning prompt suffix
SUFFIX = """
User: {query}
AI:
"""


# ================================================ #
# --------------- Paths Generation --------------- #
# ================================================ #
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator


class _SuggestedPath(BaseModel):
    path: str = Field(
        ..., description="A suggested career path based on the user's information"
    )
    reasons: list[str] = Field(
        ..., description="A list of reasons why this path is suggested"
    )


class _PathsGenerationOutput(BaseModel):
    suggested_paths: list[_SuggestedPath] = Field(
        description="A list of suggested career paths"
    )


# Paths Generation Output Parser
PATHS_GENERATION_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=_PathsGenerationOutput,
)

# Paths Generation template
PATHS_GENERATION_TEMPLATE = """
You are an AI assistant that helps people find their career path.
You have a database of job titles and their descriptions.

Offering a list of a user hard skills, and a list of a user soft skills,
you suggest a career path for the user. In addition, you provide a list of reasons why you suggested this path.

You are helpful, clever, and precise.

{format_instructions}

Here is an example:
User hard skills: {hard_skills}
User soft skills: {soft_skills}
"""

# Paths Generation prompt
PATHS_GENERATION_PROMPT = PromptTemplate(
    template=PATHS_GENERATION_TEMPLATE,
    input_variables=["hard_skills", "soft_skills"],
    partial_variables={
        "format_instructions": PATHS_GENERATION_OUTPUT_PARSER.get_format_instructions()
    },
)
