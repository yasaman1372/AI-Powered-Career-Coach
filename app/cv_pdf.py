import re
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

ACCENT = colors.HexColor("#1f5b4a")
TEXT = colors.HexColor("#222222")
MUTED = colors.HexColor("#5b6660")
BULLET = re.compile(r"^\s*(?:[-*\u2022\u25cf]|\d+[.)])\s+(.*)$")
SMART_CHARS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
}

NAME = ParagraphStyle("Name", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=ACCENT, alignment=TA_LEFT)
CONTACT = ParagraphStyle("Contact", fontName="Helvetica", fontSize=9.5, leading=13, textColor=MUTED)
HEADING = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=ACCENT, spaceBefore=12, spaceAfter=2)
BODY = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14, textColor=TEXT, spaceAfter=3)
SUBHEAD = ParagraphStyle("Subhead", parent=BODY, fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=1)


def _clean(text: str) -> str:
    """Map typographic characters and drop what the built-in PDF fonts cannot show."""

    for source, target in SMART_CHARS.items():
        text = text.replace(source, target)
    return text.encode("cp1252", errors="ignore").decode("cp1252")


def _inline(text: str) -> str:
    text = escape(_clean(text.strip()))
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _is_heading(line: str) -> bool:
    stripped = line.strip().lstrip("#").strip().rstrip(":")
    letters = [c for c in stripped if c.isalpha()]
    return bool(letters) and len(stripped) <= 40 and (
        line.lstrip().startswith("#") or all(c.isupper() for c in letters)
    )


def build_cv_pdf(text: str) -> bytes:
    """Render plain-text CV content as a clean single-column A4 PDF."""

    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)

    story: list = []
    if lines:
        story.append(Paragraph(_inline(lines.pop(0).lstrip("# ")), NAME))
    contact = []
    while lines and lines[0].strip() and not _is_heading(lines[0]) and not BULLET.match(lines[0]):
        contact.append(_inline(lines.pop(0)))
    if contact:
        story.append(Paragraph(" | ".join(contact), CONTACT))

    bullets: list[ListItem] = []

    def flush_bullets() -> None:
        if bullets:
            story.append(
                ListFlowable(bullets[:], bulletType="bullet", start="\u2022", leftIndent=12, bulletFontSize=9)
            )
            bullets.clear()

    for line in lines:
        if not line.strip():
            flush_bullets()
            continue
        match = BULLET.match(line)
        if match:
            bullets.append(ListItem(Paragraph(_inline(match.group(1)), BODY)))
            continue
        flush_bullets()
        if _is_heading(line):
            story.append(Paragraph(_inline(line.strip().lstrip("#").strip().rstrip(":")), HEADING))
            story.append(HRFlowable(width="100%", thickness=0.6, color=ACCENT, spaceAfter=4))
        elif line.strip().startswith("**") and line.strip().endswith("**"):
            story.append(Paragraph(_inline(line), SUBHEAD))
        else:
            story.append(Paragraph(_inline(line), BODY))
    flush_bullets()
    story.append(Spacer(1, 4))

    buffer = BytesIO()
    SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm, title="Tailored CV",
    ).build(story)
    return buffer.getvalue()
