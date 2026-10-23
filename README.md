# AI-Powered Career Coach

An AI career coach for mock interviews, CV feedback, career advice, personalized learning roadmaps, job search, and tailoring a CV to a selected job listing.

## Live app

[Try AI-Powered Career Coach](https://ai-powered-career-coach-beryl.vercel.app/)

## Features

- **Mock interview:** guided setup, realistic interview questions, answer evaluation, skip, and end controls.
- **CV review:** upload a PDF or DOCX to receive strengths and improvement suggestions.
- **Career advice:** get practical, personalized guidance.
- **Personalized learning roadmap:** create a step-by-step learning plan based on a target role or skill, existing experience, weekly study time, and target timeframe.
- **Job search:** search job listings from Remotive, Arbeitnow, Jobicy, Remote OK, and The Muse. Choose a role, location, and posting-time filter. Results show as linked cards with source, location, type, salary when available, and posting date.
- **CV job-match score:** optionally upload a CV from a job card to get an evidence-based match score, listed qualifications, gaps, and suggestions for that role.
- **CV tailoring:** select **Tailor my CV for this job** on a job card, upload a PDF or DOCX, and receive a draft tailored to the listing. The coach is instructed not to invent experience or qualifications.
- **CV export:** download tailored drafts as a formatted PDF or editable DOCX.
- **Job-ad links:** the coach attempts to read public job pages. Some sites, including LinkedIn, require a login or block automated access; paste the job description into chat when a link cannot be read.

Job search uses public APIs and does not require separate API keys. Listing availability and location coverage depend on the sources.

## Requirements

- Python 3.10 or newer
- An OpenAI API key

## Run locally

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-api-key
```

Start the app:

```bash
uvicorn app.index:app --reload
```

Open <http://localhost:8000>. The API health check is at <http://localhost:8000/health>, and FastAPI's interactive API documentation is at <http://localhost:8000/docs>.

The main routes are:

- `GET /` — web app
- `GET /health` — health check
- `POST /chat` — chat and job search
- `POST /chat/cv-review` — PDF or DOCX CV review and tailoring
- `POST /chat/job-match-score` — compare an uploaded PDF or DOCX CV with a selected listing
- `POST /cv/pdf` — render a tailored CV as PDF
- `POST /cv/docx` — render a tailored CV as editable DOCX

CV uploads are limited to 5 MB. The match score is an AI-generated estimate based on the information visible in the uploaded CV and job listing; it is not a hiring prediction.

## Deploy

Deploy the repository with a Python/FastAPI-compatible host. Configure the application entry point as `app.index:app`, install dependencies from `requirements.txt`, and set `OPENAI_API_KEY` in the host's environment-variable settings. Do not commit `.env` or put the API key in frontend code.

For Vercel, import the repository, configure the Python/FastAPI application entry point for `app.index:app` if it is not detected automatically, add `OPENAI_API_KEY` to the project's environment variables, and redeploy after changes.

## Project layout

```text
app/
  career_coach.py  AI prompts, responses, and tool handling
  cv_docx.py       Editable DOCX generation
  cv_documents.py  PDF/DOCX text extraction
  cv_pdf.py        Formatted PDF generation
  index.py         FastAPI routes and app entry point
  job_search.py    Public job-board search
public/
  index.html       Frontend
requirements.txt   Python dependencies
```
