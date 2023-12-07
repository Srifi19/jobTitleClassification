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

Offering a list of a user hard skills, and a list of a user soft skills, in addition to optional information about the user's education and experience,
you suggest a career path for the user. In addition, you provide a list of reasons why you suggested this path.

Not that a list of job titles may be provided to you, and you should generate job titles that are different from the provided ones.

You are helpful, clever, and precise.

{format_instructions}

Here are the info:
User hard skills: {hard_skills}
User soft skills: {soft_skills}
User education: {education}
User experience: {experience}

Paths to ignore: {ignore_titles}
"""

# Paths Generation prompt
PATHS_GENERATION_PROMPT = PromptTemplate(
    template=PATHS_GENERATION_TEMPLATE,
    input_variables=[
        "hard_skills",
        "soft_skills",
        "education",
        "experience",
        "ignore_titles",
    ],
    partial_variables={
        "format_instructions": PATHS_GENERATION_OUTPUT_PARSER.get_format_instructions()
    },
)


# ============================================================ #
# --------------- Path requirements Generation --------------- #
# ============================================================ #
class _PathRequiredField(BaseModel):
    skill: str = Field(
        ..., description="The name of the required skill (e.g. 'Basic Statistics')"
    )
    description: str = Field(
        ...,
        description="A description of the required skill (e.g. 'Get a basic understanding of statistics like mean, median, mode, variance, standard deviation, etc.')",
    )


class _PathRequirementsGenerationOutput(BaseModel):
    required_fields: list[_PathRequiredField] = Field(
        ..., description="A list of required fields"
    )


# Path required skills output parser
PATHS_REQUIREMENTS_GENERATION_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=_PathRequirementsGenerationOutput,
)


# Path required skills template
PATHS_GENERATION_TEMPLATE = """
You are an AI based career coach that helps people find their career path.

You will be given a wanted career path, a list of skills (hard skills, soft skills, education, experience).

You should generate a list of required fields (skills) for this career path.

You are helpful, clever, and precise. And you should not suggest a skill or an Education that is already in the user's list.

{format_instructions}

Here are the info:
User wanted career path: {path}
User hard skills: {hard_skills}
User soft skills: {soft_skills}
User education: {education}
User experience: {experience}
"""


# Path required skills generation prompt
PATHS_REQUIREMENTS_GENERATION_PROMPT = PromptTemplate(
    template=PATHS_GENERATION_TEMPLATE,
    input_variables=[
        "path",
        "hard_skills",
        "soft_skills",
        "education",
        "experience",
    ],
    partial_variables={
        "format_instructions": PATHS_REQUIREMENTS_GENERATION_OUTPUT_PARSER.get_format_instructions()
    },
)
