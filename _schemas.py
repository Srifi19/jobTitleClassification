from pydantic import BaseModel
from enum import Enum


# ========================================== #
# ---------- Job Title Prediction ---------- #
# ========================================== #
class Model(str, Enum):
    FAISS = "faiss"
    GPT = "gpt"
    SVC = "svc"


class PrivateJobTitle(BaseModel):
    job_title: str
    score: float | None


class GetPrivateJobTitleResponse(BaseModel):
    job_titles: list[PrivateJobTitle]
    formatted_input: str
    execution_time: float
    model: Model
    cost: dict | None = None


# ============================================== #
# ---------- Career Paths Suggestions ---------- #
# ============================================== #
class SuggestedCareerPath(BaseModel):
    path: str
    reasons: list[str]


class GetSuggestedCareerPathsResponse(BaseModel):
    suggested_paths: list[SuggestedCareerPath]
    execution_time: float
    cost: dict


# =================================================== #
# ---------- Path requirements suggestions ---------- #
# =================================================== #
class RequiredSkill(BaseModel):
    skill: str
    description: str
 
class PathRequirementsResponse(BaseModel):
    soft_skills:list[RequiredSkill]
    hard_skills:list[RequiredSkill]
    education:list[RequiredSkill]
    experience:list[RequiredSkill]
    execution_time: float
    cost: dict
    

# ========================================= #
# ---------- Courses Suggestions ---------- #
# ========================================= #
class SuggestedCourse(BaseModel):
    title:str
    description:str
    reason:str
    
class SuggestedCoursesResponse(BaseModel):
    courses:list[SuggestedCourse]
    execution_time: float
    cost: dict