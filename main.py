from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from time import time

from _schemas import GetPrivateJobTitleResponse, Model

# all models
from agents.faiss_agent import FAISS_AGENT
from agents.gpt_agent import GPT_AGENT
from agents.svc_agent import SVC_AGENT

# data loader helper
from helpers.data_loader import ALL_JOB_TITLES

# spelling corrector
from helpers.spell_checker import fix_spelling

# define the app
app = FastAPI()

# define CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify the list of allowed origins here
    allow_credentials=True,
    allow_methods=["*"],  # You can specify the list of allowed methods here
    allow_headers=["*"],  # You can specify the list of allowed headers here
)


@app.get(
    "/private-job-title",
    response_model=list[GetPrivateJobTitleResponse],
)
async def get_private_job_title(
    job_title: str,
    model: Model = Model.FAISS,
    top_n: int = 5,
    correct_spelling: bool = False,
    ignore_case: bool = True,
):
    # start timer
    start_time = time()

    # check entered job title
    if job_title.strip() == "":
        raise HTTPException(status_code=400, detail="user_job_title cannot be empty")

    # make sure that the model is valid
    if model not in Model.__members__.values():
        raise HTTPException(status_code=400, detail="Invalid model")

    # clean up the user input
    user_job_title = job_title.strip()

    # if ignore_case is True
    if ignore_case:
        user_job_title = user_job_title.lower()

    # -- validate user_job_title -- #
    # if user_job_title is empty
    if user_job_title == "":
        raise HTTPException(status_code=400, detail="user_job_title cannot be empty")

    # if user_job_title length is less than 5
    if len(user_job_title) < 5 or len(user_job_title) > 60:
        raise HTTPException(
            status_code=400,
            detail="Invalid user_job_title length, it should be between 5 and 60 characters",
        )

    # -- validate top_n -- #
    if top_n < 1 or top_n > 15:
        raise HTTPException(
            status_code=400,
            detail="Invalid top_n, it should be between 1 and 15",
        )

    # correct spelling
    if correct_spelling:
        user_job_title = fix_spelling(user_job_title)

    # ================================= #
    # ---------- FAISS Model ---------- #
    # ================================= #
    if model == Model.FAISS:
        faiss_results = FAISS_AGENT.get_similar_titles(
            user_job_title,
            top_n=top_n,
        )

        # return the results
        return [
            GetPrivateJobTitleResponse(
                formatted_input=user_job_title,
                job_title=option,
                score=score,
                model=model,
                execution_time=time() - start_time,
            )
            for option, score in faiss_results
        ]

    # =============================== #
    # ---------- SVC Model ---------- #
    # =============================== #
    if model == Model.SVC:
        # predict the job title
        predicted_job_titles = SVC_AGENT.predict_job_title(
            user_job_title,
            top_n=top_n,
        )

        # return the results
        return [
            GetPrivateJobTitleResponse(
                formatted_input=user_job_title,
                job_title=option,
                score=None,
                model=model,
                execution_time=time() - start_time,
            )
            for option in predicted_job_titles
        ]

    # ================================= #
    # ---------- GPT Model ------------ #
    # ================================= #
    gpt_response = GPT_AGENT.get_best_job_title(
        user_job_title,
        top_n=top_n,
    )
    return [
        GetPrivateJobTitleResponse(
            formatted_input=user_job_title,
            job_title=gpt_response.response,
            score=None,
            model=model,
            execution_time=time() - start_time,
            cost={
                "prompt_tokens": gpt_response.input_tokens,
                "completion_tokens": gpt_response.output_tokens,
                "cost": gpt_response.cost,
            },
        )
    ]
