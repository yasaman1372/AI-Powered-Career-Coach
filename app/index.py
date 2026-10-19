from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError
from pydantic import BaseModel, Field
from typing import Literal

from app.career_coach import (
    CareerAdviceReply,
    CareerCoach,
    CVReviewReply,
    ModeSelection,
    MockInterviewReply,
)


load_dotenv()

app = FastAPI(
    title="AI Travel Advisor API",
    description="Create personalized travel plans with an AI advisor.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

coach = CareerCoach()

class ChatRequest(BaseModel):
    """A message sent to the career coach."""

    message: str = Field(min_length=1)
    previous_response_id: str | None = None


class ChatResponse(BaseModel):
    response_id: str
    mode: Literal["mock_interview", "cv_review", "career_advice"] | None = None
    reply: ModeSelection | MockInterviewReply | CVReviewReply | CareerAdviceReply


@app.get("/", include_in_schema=False)
def home() -> dict[str, str]:
    """Point API visitors to the service documentation and health check."""

    return {
        "service": "AI Career Coach API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the HTTP service is running."""

    return {"status": "ok"}


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
            **coach.get_response(message, request.previous_response_id)
        )
    except OpenAIError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI provider could not process the request.",
        ) from error
