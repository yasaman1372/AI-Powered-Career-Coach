import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from openai import OpenAIError
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

from app.career_coach import (
    CareerAdviceReply,
    CareerCoach,
    CVReviewReply,
    JobMatchReply,
    LearningRoadmapReply,
    ModeSelection,
    MockInterviewReply,
)
from app.cv_documents import CVDocumentError, extract_cv_text
from app.cv_docx import build_cv_docx
from app.cv_pdf import build_cv_pdf
from app.job_search import JobOffer, search_jobs

logger = logging.getLogger(__name__)


load_dotenv()

app = FastAPI(
    title="AI Career Coach API",
    description="Get interview practice, CV feedback, and career advice.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

coach = CareerCoach()
MAX_CV_FILE_BYTES = 5 * 1024 * 1024

class ChatRequest(BaseModel):
    """A message sent to the career coach."""

    message: str = Field(min_length=1)
    previous_response_id: str | None = None
    mode: Literal["mock_interview", "cv_review", "career_advice", "learning_roadmap"] | None = None


class ChatResponse(BaseModel):
    response_id: str
    mode: Literal["mock_interview", "cv_review", "career_advice", "learning_roadmap"] | None = None
    reply: ModeSelection | MockInterviewReply | CVReviewReply | JobMatchReply | CareerAdviceReply | LearningRoadmapReply
    jobs: list[JobOffer] | None = None


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the chat interface from the same host as the API."""

    frontend_path = Path(__file__).resolve().parent.parent / "public" / "index.html"
    return FileResponse(frontend_path)


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the HTTP service is running."""

    return {"status": "ok"}


class CVPdfRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


@app.post("/cv/pdf", include_in_schema=False)
def cv_pdf(request: CVPdfRequest) -> Response:
    """Turn a tailored CV draft into a formatted PDF download."""

    return Response(
        content=build_cv_pdf(request.text),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="tailored-cv.pdf"'},
    )


@app.post("/cv/docx", include_in_schema=False)
def cv_docx(request: CVPdfRequest) -> Response:
    """Turn a tailored CV draft into an editable Word document."""

    return Response(
        content=build_cv_docx(request.text),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="tailored-cv.docx"'},
    )


@app.get("/jobs", response_model=list[JobOffer], response_model_exclude_none=True)
def jobs(role: str, location: str = "", max_age_days: int = 0) -> list[JobOffer]:
    """Search job boards directly, without calling the AI model."""

    if not role.strip():
        raise HTTPException(status_code=422, detail="Role must not be blank.")
    return search_jobs(role, location, max(0, max_age_days))


@app.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
def chat(request: ChatRequest) -> ChatResponse:
    """Send one message in a client-managed conversation."""

    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=422,
            detail="Message must not be blank.",
        )

    try:
        return ChatResponse(
            **coach.get_response(message, request.previous_response_id, request.mode)
        )
    except OpenAIError as error:
        logger.exception("OpenAI request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider could not process the request.",
        ) from error
    except Exception as error:
        logger.exception("Unexpected error while handling chat request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The server encountered an unexpected error. Check the server logs.",
        ) from error


@app.post(
    "/chat/cv-review",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
async def chat_with_cv(
    message: str = Form(default=""),
    previous_response_id: str | None = Form(default=None),
    target_job: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> ChatResponse:
    """Review an uploaded PDF or DOCX CV without storing it on disk."""

    message = message.strip()
    content = await file.read(MAX_CV_FILE_BYTES + 1)
    await file.close()
    if len(content) > MAX_CV_FILE_BYTES:
        raise HTTPException(status_code=413, detail="CV files must be 5 MB or smaller.")

    try:
        cv_text = extract_cv_text(file.filename or "", content)
    except CVDocumentError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        job = JobOffer.model_validate_json(target_job) if target_job else None
    except ValidationError as error:
        raise HTTPException(status_code=422, detail="The selected job listing is invalid.") from error

    try:
        return ChatResponse(
            **coach.review_cv(
                message,
                cv_text,
                file.filename or "uploaded CV",
                previous_response_id,
                job,
            )
        )
    except OpenAIError as error:
        logger.exception("OpenAI CV review request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider could not process the CV review.",
        ) from error
    except Exception as error:
        logger.exception("Unexpected error while handling CV review request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The server encountered an unexpected error. Check the server logs.",
        ) from error


@app.post(
    "/chat/job-match-score",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
async def score_job_match(
    target_job: str = Form(...),
    file: UploadFile = File(...),
) -> ChatResponse:
    """Compare an uploaded CV with a selected job listing."""

    content = await file.read(MAX_CV_FILE_BYTES + 1)
    await file.close()
    if len(content) > MAX_CV_FILE_BYTES:
        raise HTTPException(status_code=413, detail="CV files must be 5 MB or smaller.")

    try:
        cv_text = extract_cv_text(file.filename or "", content)
        job = JobOffer.model_validate_json(target_job)
    except CVDocumentError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValidationError as error:
        raise HTTPException(status_code=422, detail="The selected job listing is invalid.") from error

    try:
        return ChatResponse(
            **coach.score_job_match(cv_text, file.filename or "uploaded CV", job)
        )
    except OpenAIError as error:
        logger.exception("OpenAI job match score request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider could not calculate the job match score.",
        ) from error
    except Exception as error:
        logger.exception("Unexpected error while calculating job match score")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The server encountered an unexpected error. Check the server logs.",
        ) from error
