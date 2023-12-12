from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from time import time

from _schemas import *

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


@app.post(
    "/suggest-career-path",
    response_model=GetSuggestedCareerPathsResponse,
)
async def get_suggested_career_path(
    hard_skills: list[str],
    soft_skills: list[str],
    tools: list[str],
    education: list[str],
    experience: list[str],
    ignore_titles: list[str],
):
    # start timer
    start_time = time()

    # --- validate required params --- #
    # check entered hard skills
    if len(hard_skills) in [0, 30]:
        raise HTTPException(
            status_code=400, detail="hard_skills cannot be empty or more than 30"
        )

    # check entered soft skills
    if len(soft_skills) in [0, 15]:
        raise HTTPException(
            status_code=400, detail="soft_skills cannot be empty or more than 15"
        )

    # get suggested career paths
    suggested_paths = GPT_AGENT.generate_recommended_paths(
        hard_skills=hard_skills,
        soft_skills=soft_skills,
        tools=tools,
        education=education,
        experience=experience,
        ignore_titles=ignore_titles,
    )

    # return the results
    results = [
        SuggestedCareerPath(
            path=suggested_path.path,
            reasons=suggested_path.reasons,
        )
        for suggested_path in suggested_paths.response.suggested_paths
    ]
    return GetSuggestedCareerPathsResponse(
        suggested_paths=results,
        execution_time=time() - start_time,
        cost={
            "prompt_tokens": suggested_paths.input_tokens,
            "completion_tokens": suggested_paths.output_tokens,
            "cost": suggested_paths.cost,
        },
    )


@app.post(
    "/path-requirements",
    response_model=PathRequirementsResponse,
)
async def get_path_requirements(
    path: str,
    hard_skills: list[str],
    soft_skills: list[str],
    tools: list[str],
    education: list[str],
    experience: list[str],
):
    # start timer
    start_time = time()

    # --- validate required params --- #
    # check entered hard skills
    if len(hard_skills) in [0, 30]:
        raise HTTPException(
            status_code=400, detail="hard_skills cannot be empty or more than 30"
        )

    # check entered soft skills
    if len(soft_skills) in [0, 15]:
        raise HTTPException(
            status_code=400, detail="soft_skills cannot be empty or more than 15"
        )

    # get suggested career paths
    gpt_response = GPT_AGENT.generate_path_requirements(
        path=path,
        hard_skills=hard_skills,
        tools=tools,
        soft_skills=soft_skills,
        education=education,
        experience=experience,
    )
    data = gpt_response.response

    return PathRequirementsResponse(
        soft_skills=[
            RequiredSkill(skill=skill.skill, description=skill.description)
            for skill in data.required_soft_skills
        ],
        tools=[
            RequiredSkill(skill=tool.tool, description=tool.description)
            for tool in data.tools
        ],
        hard_skills=[
            RequiredSkill(skill=skill.skill, description=skill.description)
            for skill in data.required_hard_skills
        ],
        education=[
            RequiredSkill(skill=education.education, description=education.description)
            for education in data.Education
        ],
        experience=[
            RequiredSkill(
                skill=experience.experience, description=experience.description
            )
            for experience in data.Experience
        ],
        execution_time=time() - start_time,
        cost={
            "prompt_tokens": gpt_response.input_tokens,
            "completion_tokens": gpt_response.output_tokens,
            "cost": gpt_response.cost,
        },
    )


@app.post("/suggested-courses", response_model=SuggestedCoursesResponse)
async def get_suggested_courses(
    needed_skill: str,
    hard_skills: list[str],
    tools: list[str],
):
    # start timer
    start_time = time()

    # check entered needed_skill
    if needed_skill.strip() == "":
        raise HTTPException(status_code=400, detail="user_job_title cannot be empty")

    # validate hard_skills list
    if 15 < len(hard_skills) == 0:
        raise HTTPException(
            status_code=400,
            detail="Hard skills list cannot be empty or contain more than 15 element",
        )

    # validate tools list
    if 15 < len(tools) == 0:
        raise HTTPException(
            status_code=400,
            detail="Tools list cannot be empty or contain more than 15 element",
        )

    # get GPT response
    gpt_response = GPT_AGENT.recommend_courses(
        topic=needed_skill, hard_skills=hard_skills, tools=tools
    )

    # return the output
    return SuggestedCoursesResponse(
        courses=[
            SuggestedCourse(
                title=s["title"],
                description=s["description"],
                reason=s["reason"],
                link=s["link"],
            )
            for s in gpt_response.response
        ],
        execution_time=time() - start_time,
        cost={
            "prompt_tokens": gpt_response.input_tokens,
            "completion_tokens": gpt_response.output_tokens,
            "cost": gpt_response.cost,
        },
    )


@app.get(
    "/private-job-title",
    response_model=GetPrivateJobTitleResponse,
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

        return GetPrivateJobTitleResponse(
            job_titles=[
                PrivateJobTitle(
                    job_title=option,
                    score=score,
                )
                for option, score in faiss_results
            ],
            formatted_input=user_job_title,
            execution_time=time() - start_time,
            model=model,
            cost=None,
        )

    # =============================== #
    # ---------- SVC Model ---------- #
    # =============================== #
    if model == Model.SVC:
        # predict the job title
        predicted_job_title = SVC_AGENT.predict_job_title(
            user_job_title,
            top_n=top_n,
        )

        return GetPrivateJobTitleResponse(
            job_titles=[
                PrivateJobTitle(
                    job_title=predicted_job_title,
                    score=None,
                )
            ],
            formatted_input=user_job_title,
            execution_time=time() - start_time,
            model=model,
            cost=None,
        )

    # ================================= #
    # ---------- GPT Model ------------ #
    # ================================= #
    gpt_response = GPT_AGENT.get_best_job_title(
        user_job_title,
        top_n=top_n,
    )
    return GetPrivateJobTitleResponse(
        job_titles=[
            PrivateJobTitle(
                formatted_input=user_job_title,
                job_title=gpt_response.response,
                score=None,
            )
        ],
        formatted_input=user_job_title,
        execution_time=time() - start_time,
        model=model,
        cost={
            "prompt_tokens": gpt_response.input_tokens,
            "completion_tokens": gpt_response.output_tokens,
            "cost": gpt_response.cost,
        },
    )
