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

Offering a list of user hard skills, a list of user soft skills and a list of user tools, in addition to optional information about the user's education and experience,
you suggest a cutting-edge career path for the user. In addition, you provide a list of reasons why you suggested this path.

Note that a list of job titles may be provided to you, and you should generate job titles that are different from the provided ones.

You are helpful, clever, and precise.

{format_instructions}

Here are the info:
User tools: {tools}
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
        "tools",
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
class _skillRequired(BaseModel):
    skill: str = Field(
        ..., description="The name of the required skill (e.g. 'Basic Statistics')"
    )
    description: str = Field(
        ...,
        description="A description of the required skill (e.g. 'Get a basic understanding of statistics like mean, median, mode, variance, standard deviation, etc.')",
    )


class _educationRequired(BaseModel):
    education: str = Field(
        ..., description="The required education, including the name and type of degree"
    )
    description: str = Field(
        ...,
        description="A description of the required education and what one may learn",
    )


class _experienceRequired(BaseModel):
    experience: str = Field(
        ...,
        description="The required experience, specifying the position and type of experiences",
    )
    description: str = Field(
        ...,
        description="A description of the required experience and what one may learn",
    )


class _toolRequired(BaseModel):
    tool: str = Field(..., description="The name of the required tool (e.g. 'Java')")
    description: str = Field(
        ...,
        description="A description of the required tool Java (e.g., 'Syntax, Concepts etc...')",
    )


class _PathRequirementsGenerationOutput(BaseModel):
    required_hard_skills: list[_skillRequired] = Field(
        ..., description="A list of required hard skills"
    )
    required_soft_skills: list[_skillRequired] = Field(
        ..., description="A list of required soft skills"
    )
    tools: list[_toolRequired] = Field(..., description="A list of required tools")
    Education: list[_educationRequired] = Field(
        ..., description="A list of required educations"
    )
    Experience: list[_experienceRequired] = Field(
        ..., description="A list of required experiences"
    )


# Path required skills output parser
PATHS_REQUIREMENTS_GENERATION_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=_PathRequirementsGenerationOutput,
)


# Path required skills template
PATHS_GENERATION_TEMPLATE = """


Given the user's desired career path in {path} and their existing set of skills, tools, education, and experience, your task is to generate a list of additional required fields (skills, education, experience). Be helpful, clever, and precise. Avoid repeating anything the user has given that is already in the user's list, and ensure there are no repeated skills.

Provide recommendations for necessary skills, education,tools and experiences, taking into account the user's current qualifications. Additionally, ensure that education is only suggested if it is not already included in the user's provided education. For experience, suggest relevant and sensible additions.

{format_instructions}

These Are the given info:
User wanted career path: {path}
User hard skills: {hard_skills}
User soft skills: {soft_skills}
User tools: {tools}
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
        "tools",
        "education",
        "experience",
    ],
    partial_variables={
        "format_instructions": PATHS_REQUIREMENTS_GENERATION_OUTPUT_PARSER.get_format_instructions()
    },
)


# ================================================== #
# --------------- Courses Suggestion --------------- #
# ================================================== #
class _SuggestedCourse(BaseModel):
    course_id: str = Field(..., description="The id of the suggested course")
    suggestion_reason: str = Field(
        ..., description="The reason why this course is suggested"
    )


class _CoursesSuggestionResponse(BaseModel):
    suggested_courses: list[_SuggestedCourse] = Field(
        ..., description="A list of suggested courses"
    )


# suggested course output parser
SUGGESTED_COURSE_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=_CoursesSuggestionResponse,
)

# suggested course template
SUGGESTED_COURSE_TEMPLATE = """
You are an AI assistant that helps people choose the most suitable courses for them.
You will be given a list of courses and you should choose only the most suitable courses for the user.
To help you decide which courses be suit the user, you will be given the career goal, a list of user hard skills, and list of user tools. 

You are helpful, clever, and precise. And you should be able to choose only the most suitable courses for the user with maximum of 3 courses

{format_instructions}

These Are the given info:
List of courses: {courses}
User career goal: {career_goal}
User hard skills: {hard_skills}
User tools: {tools}
"""


# suggested course generation prompt
SUGGESTED_COURSE_PROMPT = PromptTemplate(
    template=SUGGESTED_COURSE_TEMPLATE,
    input_variables=[
        "courses",
        "career_goal",
        "hard_skills",
        "tools",
    ],
    partial_variables={
        "format_instructions": SUGGESTED_COURSE_OUTPUT_PARSER.get_format_instructions()
    },
)
