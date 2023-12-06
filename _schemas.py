from pydantic import BaseModel
from enum import Enum


class Model(str, Enum):
    FAISS = "faiss"
    GPT = "gpt"
    SVC = "svc"


class GetPrivateJobTitleResponse(BaseModel):
    formatted_input: str
    job_title: str
    score: float | None
    execution_time: float
    model: Model
    cost: dict | None = None


class SuggestedCareerPath(BaseModel):
    path: str
    reasons: list[str]


class GetSuggestedCareerPathsResponse(BaseModel):
    suggested_paths: list[SuggestedCareerPath]
    execution_time: float
    cost: dict
