import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import zip_longest

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
JOBICY_URL = "https://jobicy.com/api/v2/remote-jobs"
REMOTEOK_URL = "https://remoteok.com/api"
MUSE_URL = "https://www.themuse.com/api/public/jobs"
REQUEST_TIMEOUT = 8
MAX_RESULTS = 10
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
    """Return a score, or None unless every search term appears in the job title.

    Matching on the title only avoids "data scientist" accepting a job titled
    "Scientist" just because "data" appears in its description.
    """

    title, other = title.lower(), other.lower()
    if not terms:
        return 0
    if not all(term in title for term in terms):
        return None
    return sum(2 + (term in other) for term in terms)


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


def _recent_enough(posted: str | None, max_age_days: int) -> bool:
    if not max_age_days or not posted:
        return True
    try:
        date = datetime.strptime(posted[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - date).days <= max_age_days


def _iso_date(value: str | int | None) -> str | None:
    if not value:
        return None
    if str(value).isdigit():
        return datetime.fromtimestamp(int(value), tz=timezone.utc).strftime("%Y-%m-%d")
    return str(value)[:10]


def _search_jobicy(query: str, location: str) -> list[JobOffer]:
    response = requests.get(
        JOBICY_URL, params={"count": 100}, headers=HEADERS, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    terms = _terms(query)
    scored = []
    for job in response.json().get("jobs", []):
        job_location = job.get("jobGeo") or "Anywhere"
        wanted = location.strip().lower()
        if wanted and wanted not in REMOTE_LOCATION_WORDS:
            if wanted not in job_location.lower() and "anywhere" not in job_location.lower():
                continue
        other = " ".join(
            [" ".join(job.get("jobIndustry") or []), job.get("jobExcerpt") or ""]
        )
        score = _relevance(terms, job.get("jobTitle") or "", other)
        if score is None:
            continue
        scored.append(
            (
                score,
                JobOffer(
                    title=(job.get("jobTitle") or "").strip(),
                    company=(job.get("companyName") or "").strip(),
                    location=f"Remote - {job_location}",
                    url=job.get("url", ""),
                    source="Jobicy",
                    job_type=", ".join(job.get("jobType") or []) or None,
                    posted=_iso_date(job.get("pubDate")),
                    description=_description_text(job.get("jobDescription") or ""),
                ),
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [offer for _, offer in scored]


def _search_remoteok(query: str, location: str) -> list[JobOffer]:
    response = requests.get(REMOTEOK_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    terms = _terms(query)
    wanted = location.strip().lower()
    scored = []
    for job in response.json():
        if not isinstance(job, dict) or not job.get("position"):
            continue
        job_location = job.get("location") or "Worldwide"
        if wanted and wanted not in REMOTE_LOCATION_WORDS:
            if wanted not in job_location.lower() and job.get("location"):
                continue
        other = " ".join([" ".join(job.get("tags") or []), (job.get("description") or "")[:1500]])
        score = _relevance(terms, job["position"], other)
        if score is None:
            continue
        salary = None
        if job.get("salary_min") and job.get("salary_max"):
            salary = f"${job['salary_min']} - ${job['salary_max']}"
        scored.append(
            (
                score,
                JobOffer(
                    title=job["position"].strip(),
                    company=(job.get("company") or "").strip(),
                    location=f"Remote - {job_location}",
                    url=job.get("url") or job.get("apply_url") or "",
                    source="Remote OK",
                    salary=salary,
                    posted=_iso_date(job.get("epoch") or job.get("date")),
                    description=_description_text(job.get("description") or ""),
                ),
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [offer for _, offer in scored]


def _search_muse(query: str, location: str) -> list[JobOffer]:
    wanted = location.strip()
    params: dict[str, str | int] = {"page": 1}
    if wanted and wanted.lower() not in REMOTE_LOCATION_WORDS + ("anywhere",):
        params["location"] = wanted
    else:
        params["location"] = "Flexible / Remote"
    terms = _terms(query)
    scored = []
    # The Muse has no keyword search, so scan a few pages and filter locally.
    for page in (1, 2, 3):
        params["page"] = page
        response = requests.get(MUSE_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        for job in response.json().get("results", []):
            title = job.get("name") or ""
            categories = " ".join(c.get("name", "") for c in job.get("categories") or [])
            other = " ".join([categories, (job.get("contents") or "")[:1500]])
            score = _relevance(terms, title, other)
            if score is None:
                continue
            locations = ", ".join(l.get("name", "") for l in job.get("locations") or [])
            scored.append(
                (
                    score,
                    JobOffer(
                        title=title.strip(),
                        company=(job.get("company") or {}).get("name", ""),
                        location=locations or "Not specified",
                        url=(job.get("refs") or {}).get("landing_page", ""),
                        source="The Muse",
                        job_type=", ".join(l.get("name", "") for l in job.get("levels") or []) or None,
                        posted=_iso_date(job.get("publication_date")),
                        description=_description_text(job.get("contents") or ""),
                    ),
                )
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [offer for _, offer in scored]


def search_jobs(query: str, location: str = "", max_age_days: int = 0) -> list[JobOffer]:
    """Search several free job boards; a failing source is skipped.

    max_age_days > 0 keeps only offers posted within that many days.
    """

    query = query.strip()
    if not query:
        return []

    def run(search) -> list[JobOffer]:
        try:
            return search(query, location)
        except Exception:
            return []

    sources = (
        _search_arbeitnow,
        _search_muse,
        _search_remotive,
        _search_jobicy,
        _search_remoteok,
    )
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        found = list(pool.map(run, sources))

    results: list[JobOffer] = []
    seen: set[str] = set()
    for group in zip_longest(*found):
        for offer in group:
            if offer and offer.url and offer.url not in seen and _recent_enough(offer.posted, max_age_days):
                seen.add(offer.url)
                results.append(offer)
    return results[:MAX_RESULTS]
