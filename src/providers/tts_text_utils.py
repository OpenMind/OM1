import re

_TTS_CORRECTIONS = [
    # Month abbreviations
    (re.compile(r"\bJan\b"), "January"),
    (re.compile(r"\bFeb\b"), "February"),
    (re.compile(r"\bMar\b"), "March"),
    (re.compile(r"\bApr\b"), "April"),
    (re.compile(r"\bJun\b"), "June"),
    (re.compile(r"\bJul\b"), "July"),
    (re.compile(r"\bAug\b"), "August"),
    (re.compile(r"\bSep(?:t)?\b"), "September"),
    (re.compile(r"\bOct\b"), "October"),
    (re.compile(r"\bNov\b"), "November"),
    (re.compile(r"\bDec\b"), "December"),
    # Address abbreviations
    (re.compile(r"\bSt\b\.?"), "Street"),
    (re.compile(r"\bAve\b\.?"), "Avenue"),
    (re.compile(r"\bBlvd\b\.?"), "Boulevard"),
    (re.compile(r"\bDr\b\.?"), "Drive"),
    (re.compile(r"\bRd\b\.?"), "Road"),
    (re.compile(r"\bLn\b\.?"), "Lane"),
    (re.compile(r"\bCt\b\.?"), "Court"),
    (re.compile(r"\bPl\b\.?"), "Place"),
    (re.compile(r"\bPkwy\b\.?"), "Parkway"),
    (re.compile(r"\bHwy\b\.?"), "Highway"),
    # Directional abbreviations
    (re.compile(r"\bN\b\.?(?=\s+[A-Z])"), "North"),
    (re.compile(r"\bS\b\.?(?=\s+[A-Z])"), "South"),
    (re.compile(r"\bE\b\.?(?=\s+[A-Z])"), "East"),
    (re.compile(r"\bW\b\.?(?=\s+[A-Z])"), "West"),
]

# Time patterns: "11:00 a.m." -> "11 a.m.", "3:30 p.m." -> "3 30 p.m."
_TIME_ON_HOUR = re.compile(r"\b(\d{1,2}):00\b")
_TIME_WITH_MINUTES = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def normalize_tts_text(text: str) -> str:
    """Expand abbreviations and reformat times for cleaner TTS output."""
    # Fix times: "11:00" -> "11", "3:30" -> "3 30"
    text = _TIME_ON_HOUR.sub(r"\1", text)
    text = _TIME_WITH_MINUTES.sub(r"\1 \2", text)

    for pattern, replacement in _TTS_CORRECTIONS:
        text = pattern.sub(replacement, text)

    # Strip non-English characters (keep ASCII letters, digits, punctuation, whitespace)
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    return text
