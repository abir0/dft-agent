from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from pydantic import HttpUrl, TypeAdapter


def ensure_https(url: str | None) -> HttpUrl | None:
    if not url:
        return None
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme in ["", "http"]:
            parsed_url = parsed_url._replace(scheme="https")
        return TypeAdapter(HttpUrl).validate_python(urlunparse(parsed_url))
    except (AttributeError, ValueError):
        return url


def add_query_params(url: str, params: dict) -> str:
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    query_params.update(params)
    updated_query = urlencode(query_params, doseq=True)
    updated_url = urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            updated_query,
            parsed_url.fragment,
        )
    )
    return updated_url
