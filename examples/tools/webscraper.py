import asyncio
from logging import getLogger
from typing import Optional, Dict, List, Tuple, Union
from playwright.async_api import async_playwright, Browser, Page, TimeoutError
from aiavatar.sts.llm import Tool

logger = getLogger(__name__)


class WebScraper:
    def __init__(self, default_user_agent: str = None, debug: bool = False):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._lock = asyncio.Lock()
        self.default_user_agent = default_user_agent or "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
        self.debug = debug

    async def initialize(self):
        if self._playwright is None or self._browser is None:
            async with self._lock:
                if self._playwright is None:
                    self._playwright = await async_playwright().start()
                if self._browser is None:
                    self._browser = await self._playwright.chromium.launch(headless=True)

    async def _wait_until_fully_rendered(self, page: Page, max_attempts: int = 5, delay_ms: int = 200):
        for attempt in range(max_attempts):
            await page.wait_for_function("document.readyState === 'complete'", timeout=10000)
            await page.wait_for_timeout(delay_ms)

            try:
                await page.wait_for_function("document.readyState === 'complete'", timeout=3000)
                return
            except TimeoutError:
                continue
        
        logger.warning(f"ReadyState is not completed for: {page.url}")

    async def fetch_body(self, url: Union[str, List[str]], headers: Optional[Dict[str, str]] = None) -> Union[dict, List[dict]]:
        if isinstance(url, list):
            return await self.fetch_multiple_bodies([(u, headers) for u in url])

        await self.initialize()
        context = await self._browser.new_context(extra_http_headers=headers or {"User-Agent": self.default_user_agent})
        page = await context.new_page()

        try:
            if self.debug:
                logger.info(f"Fetching: {url}")
            response = await page.goto(url, timeout=30000)

            if response is None:
                raise ValueError("No response returned")

            status_code = response.status

            await self._wait_until_fully_rendered(page)

            try:
                body = await page.inner_text("body")
            except Exception as ex:
                logger.warning(f"Could not get body from {url}: {ex}")
                body = ""

            return {
                "url": url,
                "status_code": status_code,
                "body": body
            }

        except Exception as ex:
            logger.error(f"Failed to fetch {url}: {ex}")
            return {
                "url": url,
                "status_code": None,
                "body": ""
            }

        finally:
            await context.close()


    async def fetch_multiple_bodies(self, requests: List[Tuple[str, Optional[Dict[str, str]]]]) -> List[dict]:
        async def fetch(url, headers):
            return await self.fetch_body(url, headers)

        tasks = [fetch(url, headers) for url, headers in requests]
        return await asyncio.gather(*tasks)

    async def shutdown(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()


class WebScraperTool(Tool):
    def __init__(
        self,
        *,
        default_user_agent: str = None,
        name=None,
        spec=None,
        instruction = None,
        is_dynamic = False,
        debug: bool = False
    ):
        self.web_scraper = WebScraper(
            default_user_agent=default_user_agent,
            debug=debug
        )

        super().__init__(
            name or "get_webpage_body",
            spec or {
                "type": "function",
                "function": {
                    "name": name or "get_webpage_body",
                    "description": "Fetch and return the body content of a web page given its URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of URLs to scrape"
                            }
                        }
                    }
                }
            },
            self.web_scraper.fetch_body,
            instruction,
            is_dynamic
        )
