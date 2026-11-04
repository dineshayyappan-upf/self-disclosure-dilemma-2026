"""Recorded statistical-text cleaning; original responses remain unchanged."""
from __future__ import annotations
import re
from typing import Any

GROUP_ALIASES = {
    "adhd", "attention deficit hyperactivity disorder", "autism", "autistic",
    "bipolar", "bipolar disorder", "bisexual", "blind", "blindness",
    "chronic pain", "clinical anxiety", "dementia", "depressed", "depression",
    "eating disorder", "elderly", "gay", "homeless", "homelessness",
    "immigrant", "immigrants", "lesbian", "mobility impairment",
    "racial minority", "refugee", "refugees", "religious minority",
    "schizophrenia", "schizophrenic", "substance use disorder", "transgender",
    "unhoused", "union member",
}


REASONING_RE = re.compile(r"<(?:reasoning|think)>.*?</(?:reasoning|think)>", re.I | re.S)


MARKDOWN_LINK_RE = re.compile(r"!?\[([^\]]+)\]\([^)]*\)")


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


HTML_RE = re.compile(r"<[^>]+>")


SPACE_RE = re.compile(r"\s+")


def remove_literal(text: str, phrase: str) -> str:
    """Remove an exact prompt/disclosure echo, ignoring letter case."""
    phrase = str(phrase or "").strip().strip('"')
    if len(phrase) < 5:
        return text
    return re.sub(re.escape(phrase), " ", text, flags=re.I)


def build_group_name_regex(records: list[dict[str, Any]]) -> re.Pattern[str]:
    """Build one inspectable regex for direct experimental-label masking."""
    names = set(GROUP_ALIASES)
    for row in records:
        if row["condition"] != "bare":
            names.add(row["condition"].replace("_", " "))
    names.discard("")
    names.discard("subclinical")

    # Longest first prevents a short label from fragmenting a longer label.
    ordered = sorted(names, key=lambda value: (-len(value.split()), -len(value)))
    alternatives = [
        re.escape(value).replace(r"\ ", r"[\s\-_]+") for value in ordered
    ]
    return re.compile(r"(?i)(?<![\w])(?:" + "|".join(alternatives) + r")(?![\w])")


def clean_response(row: dict[str, Any], group_names: re.Pattern[str]) -> str:
    """Return the lower-case text used for TF-IDF; preserve original text."""
    text = REASONING_RE.sub(" ", row.get("response", ""))
    text = remove_literal(text, row.get("query_text", ""))
    text = remove_literal(text, row.get("disclosure_text", ""))
    text = MARKDOWN_LINK_RE.sub(r" \1 ", text)
    text = URL_RE.sub(" URL ", text)
    text = HTML_RE.sub(" ", text)
    text = group_names.sub(" ", text)
    text = text.replace("_", " ")
    return SPACE_RE.sub(" ", text).strip().lower()



TYPOGRAPHY_TRANSLATION = str.maketrans({
    "’": "'",
    "‘": "'",
    "‛": "'",
    "–": "-",
    "—": "-",
    "−": "-",
})


def normalized_clean_text(
    row: dict[str, Any], group_names: re.Pattern[str]
) -> str:
    """Create the statistical text while preserving the original response.

    ``clean_response`` removes prompt text, disclosure text, direct
    condition/group names, hidden reasoning markup, URLs, HTML, and markdown
    link syntax. It also lowercases the text. We then normalize apostrophes and
    dashes so typography cannot create a spurious model difference.
    """
    text = clean_response(row, group_names)
    text = text.translate(TYPOGRAPHY_TRANSLATION)
    return re.sub(r"\s+", " ", text).strip()



VISIBLE_WORD_RE = re.compile(r"(?u)\b\w+(?:[’']\w+)*\b")


def visible_response_text(response: str) -> str:
    """Return the text a user could read, with hidden reasoning removed.

    Unlike the TF-IDF copy, this function does *not* remove question echoes,
    disclosure language, condition names, or ordinary Markdown labels.  Those
    words occupy space in the answer and therefore belong in a visible-length
    audit.  Link destinations are replaced by one ``URL`` token so a long URL
    cannot masquerade as many words.
    """
    text = REASONING_RE.sub(" ", str(response or ""))
    text = MARKDOWN_LINK_RE.sub(r" \1 ", text)
    text = URL_RE.sub(" URL ", text)
    text = HTML_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def visible_word_count(response: str) -> int:
    """Count user-visible word tokens under the documented token rule."""
    return len(VISIBLE_WORD_RE.findall(visible_response_text(response)))
