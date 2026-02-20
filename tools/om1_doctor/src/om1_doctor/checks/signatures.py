from __future__ import annotations
import re

SIGNATURES = [
    (
        "PERM_DENIED",

        "Permission denied",
        r"permission denied|access is denied",
        "Check file permissions / run with appropriate privileges.",
    ),
    (
        "MISSING_ENV",
        "Missing required environment variable",
        r"(missing|required).*(environment variable|\bENV\b)|environment variable.*(not set|unset|missing|required)",
        "Ensure required environment variables are set in your shell/service config.",
    ),
]

def match_signatures(text: str):
    hits = []
    for sid, title, pat, hint in SIGNATURES:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append({"id": sid, "title": title, "hint": hint})
    return hits

