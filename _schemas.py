from pydantic import BaseModel
from enum import Enum


class Model(str, Enum):
    FAISS = "faiss"
    GPT = "gpt"
    SVC = "svc"


class GetPrivateJobTitleRequest(BaseModel):
    user_job_title: str
    model: Model = Model.FAISS

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "user_job_title": "JobTitle",
                    "model": "faiss",
                }
            ]
        }


class GetPrivateJobTitleResponse(BaseModel):
    job_title: str
    score: float
    execution_time: float
    model: Model
    cost: dict | None = None
