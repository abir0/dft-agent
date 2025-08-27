import re
from typing import Optional
from unicodedata import normalize


def extract_match(string: str, pattern: str, index: int = 0) -> str | None:
    match_object = re.search(pattern, string)
    try:
        return match_object.group(index)
    except AttributeError:
        return None


def extract_number(
    string: str, pattern: Optional[str] = None, dtype: callable = int
) -> str:
    if not pattern:
        string = str(string).replace(",", "")  # remove commas from digits
        pattern = r"(\d*\.?\d+)"  # digits including floating points
    match_object = extract_match(string, pattern, 1)
    return dtype(match_object) if match_object else dtype(0)


def normalize_text(string: str) -> dict:
    # Unicode sanitization
    return normalize("NFKD", string.decode("utf-8"))
