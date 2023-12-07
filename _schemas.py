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
