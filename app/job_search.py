import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import zip_longest

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
REQUEST_TIMEOUT = 8
MAX_RESULTS = 6
REMOTE_LOCATION_WORDS = ("remote", "worldwide", "anywhere")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CareerCoachBot/1.0)"}


class JobOffer(BaseModel):
    title: str
    company: str
    location: str
    url: str
    source: str
    job_type: str | None = None
    salary: str | None = None
    posted: str | None = None
    description: str | None = None


def _description_text(value: str, limit: int = 4000) -> str | None:
    text = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return text[:limit] or None


def _terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[\w+#.-]+", query.lower()) if len(term) > 1]


def _relevance(terms: list[str], title: str, other: str) -> int | None:
    """Return a score, or None when some search term appears nowhere in the job."""

    title, other = title.lower(), other.lower()
    if not terms:
        return 0
    if not any(term in title for term in terms):
        return None
    if not all(term in title or term in other for term in terms):
        return None
    return sum(2 * (term in title) + (term in other) for term in terms)


def _matches_location(job_location: str, location: str) -> bool:
    wanted = location.strip().lower()
    if not wanted or wanted in REMOTE_LOCATION_WORDS:
        return True
    return wanted in job_location.lower()


def _search_remotive(query: str, location: str) -> list[JobOffer]:
    response = requests.get(
        REMOTIVE_URL,
        params={"search": query, "limit": 100},
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    terms = _terms(query)
    wanted = location.strip().lower()
    scored = []
    for job in response.json().get("jobs", []):
        job_location = job.get("candidate_required_location") or "Remote"
        if wanted and wanted not in REMOTE_LOCATION_WORDS:
            remote_ok = any(w in job_location.lower() for w in REMOTE_LOCATION_WORDS)
            if wanted not in job_location.lower() and not remote_ok:
                continue
        other = " ".join(
            [
                job.get("category") or "",
                " ".join(job.get("tags") or []),
                (job.get("description") or "")[:1500],
            ]
        )
        score = _relevance(terms, job.get("title") or "", other)
        if score is None:
            continue
        scored.append(
            (
                score,
                JobOffer(
                    title=(job.get("title") or "").strip(),
                    company=(job.get("company_name") or "").strip(),
                    location=f"Remote - {job_location}",
                    url=job.get("url", ""),
                    source="Remotive",
                    job_type=(job.get("job_type") or "").replace("_", " ") or None,
                    salary=job.get("salary") or None,
                    posted=(job.get("publication_date") or "")[:10] or None,
                    description=_description_text(job.get("description") or ""),
                ),
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [offer for _, offer in scored]


def _search_arbeitnow(query: str, location: str) -> list[JobOffer]:
    response = requests.get(ARBEITNOW_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    terms = _terms(query)
    scored = []
    for job in response.json().get("data", []):
        job_location = job.get("location") or ""
        if not _matches_location(job_location, location):
            continue
        other = " ".join(
            [" ".join(job.get("tags") or []), (job.get("description") or "")[:1500]]
        )
        score = _relevance(terms, job.get("title") or "", other)
        if score is None:
            continue
        posted = None
        if job.get("created_at"):
            posted = datetime.fromtimestamp(
                int(job["created_at"]), tz=timezone.utc
            ).strftime("%Y-%m-%d")
        scored.append(
            (
                score,
                JobOffer(
                    title=(job.get("title") or "").strip(),
                    company=(job.get("company_name") or "").strip(),
                    location="Remote" if job.get("remote") else job_location or "Not specified",
                    url=job.get("url", ""),
                    source="Arbeitnow",
                    job_type=", ".join(job.get("job_types") or []) or None,
                    posted=posted,
                    description=_description_text(job.get("description") or ""),
                ),
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [offer for _, offer in scored]


def search_jobs(query: str, location: str = "") -> list[JobOffer]:
    """Search Remotive and Arbeitnow for job offers; a failing source is skipped."""

    query = query.strip()
    if not query:
        return []

    def run(search) -> list[JobOffer]:
        try:
            return search(query, location)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=2) as pool:
        remotive, arbeitnow = pool.map(run, (_search_remotive, _search_arbeitnow))

    results: list[JobOffer] = []
    seen: set[str] = set()
    for pair in zip_longest(remotive, arbeitnow):
        for offer in pair:
            if offer and offer.url and offer.url not in seen:
                seen.add(offer.url)
                results.append(offer)
    return results[:MAX_RESULTS]
