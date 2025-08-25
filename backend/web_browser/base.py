from itertools import cycle
from typing import Any, Dict, Optional

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.exceptions import RateLimitException, RequestException
from backend.logger import logger


class AsyncClient:
    def __init__(
        self,
        user_agent: Optional[str] = None,
        headers: Dict[str, str] = {},
        cookies: Dict[str, str] = {},
        rate_limit: int = 10,
        timeout: int = 10,
    ):
        # Rotating user agents for anti-blocking
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:108.0) Gecko/20100101 Firefox/108.0",
        ]
        self.user_agents = cycle(user_agents)
        self.user_agent = user_agent
        self.headers = headers
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.cookies = cookies
        self.client = httpx.AsyncClient(
            headers=self._get_headers(),
            cookies=self._get_cookies(),
            follow_redirects=True,
            timeout=httpx.Timeout(self.timeout),
        )

    async def __aenter__(self):
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self.client.__aexit__(*args)

    def _get_next_user_agent(self) -> Optional[str]:
        if self.user_agent:
            return self.user_agent
        return next(self.user_agents)

    def _get_headers(self) -> Dict[str, str]:
        """Generate headers with random user agent."""
        self.headers["User-Agent"] = self._get_next_user_agent()
        return self.headers

    def _get_cookies(self) -> Dict[str, str]:
        return self.cookies

    def _is_blocked(self, response: httpx.Response) -> bool:
        if response.status_code in {403, 429}:
            return True

        soup = BeautifulSoup(response.text, "html.parser")
        if soup.find("div", id="captcha"):
            return True

        return False

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, RateLimitException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            response = await self.client.request(method, url, **kwargs)
            logger.debug(f"Request to {url} returned {response.status_code}")

            if self._is_blocked(response):
                logger.warning("Blocked detected on request, attempting again")
                self.client.cookies.clear()

            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e}")
            if e.response.status_code in {403, 429}:
                logger.warning("Blocked by website, rotating user agent")
                self.client.cookies.clear()
            raise RequestException(f"HTTP error {e.response.status_code}: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise RequestException(f"Request failed: {e}")

    async def get_html(self, url: str, **kwargs) -> str:
        """Fetch the HTML content of the given URL."""
        response = await self.request("GET", url, **kwargs)
        return response.text

    async def get_json(self, url: str, **kwargs) -> Dict[str, Any]:
        """Fetch and return JSON content from the given URL."""
        response = await self.request("GET", url, **kwargs)
        return response.json()
