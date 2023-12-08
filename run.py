from helpers.logger import LOGGER, LoggingLevels

LOGGER.log("Starting app...", level=LoggingLevels.INFO)

# setup OPENAI_API_KEY
from os import environ

environ["OPENAI_API_KEY"] = "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"

# -1- Load faiss agent
from agents.faiss_agent import FAISS_AGENT

# -2- Load gpt agent
from agents.gpt_agent import GPT_AGENT

# -3- Load svc agent
from agents.svc_agent import SVC_AGENT

# -4- Load all data
from helpers.data_loader import ALL_JOB_TITLES

# detect platform
import platform

PLATFORM = 1 if platform.system() == "Windows" else 2

if __name__ == "__main__":
    from helpers.logger import LOGGER, LoggingLevels

    # start uvicorn server
    LOGGER.log("Starting uvicorn server...", level=LoggingLevels.INFO)

    # windows
    if PLATFORM == 1:
        from os import system

        system("cmd /k uvicorn main:app --reload")

    # linux
    else:
        # Run FastAPI app using uvicorn by providing the import string
        import uvicorn

        uvicorn_cmd = "main:app"
        uvicorn.run(uvicorn_cmd, host="0.0.0.0", port=8000, reload=True)
