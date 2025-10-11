import asyncio
from logging import getLogger
from typing import Optional, Dict, List, Tuple, Union
from urllib.parse import urlparse, parse_qs
from playwright.async_api import async_playwright, Browser, Page, TimeoutError
from aiavatar.sts.llm import Tool

logger = getLogger(__name__)


class WebScraper:
    def __init__(self, *, default_user_agent: str = None, openai_api_key: str = None, openai_base_url: str = None, openai_model: str = "gpt-4.1", return_summary: bool = False, summary_system_prompt: str = None, debug: bool = False):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._lock = asyncio.Lock()
        self.default_user_agent = default_user_agent or "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"

        if openai_api_key:
            import openai
            self.openai_model = openai_model
            self.summary_system_prompt = summary_system_prompt or "Summarize the given text in up to 500 characters."
            if "azure" in self.openai_model:
                api_version = parse_qs(urlparse(openai_base_url).query).get("api-version", [None])[0]
                self.openai_client = openai.AsyncAzureOpenAI(
                    api_key=openai_api_key,
                    api_version=api_version,
                    base_url=openai_base_url,
                    timeout=30000
                )
            else:
                self.openai_client = openai.AsyncClient(api_key=openai_api_key, base_url=openai_base_url, timeout=30000)
        else:
            self.openai_client = None

        self.return_summary = return_summary
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

    async def _make_summary(self, body: str, research_goal: str = None) -> str:
        try:
            system_prompt = self.summary_system_prompt
            if research_goal:
                system_prompt += f"\nConsider the following goal when summarizing, as the summary should focus on collecting information relevant to achieving it:\n{research_goal}"

            resp = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": body}
                ],
            )

            return resp.choices[0].message.content

        except Exception as ex:
            logger.error(f"Error at _make_summary. Return original body instead.: {ex}")
            return body

    async def fetch_body(self, url: Union[str, List[str]], headers: Optional[Dict[str, str]] = None, research_goal: str = None) -> Union[dict, List[dict]]:
        if isinstance(url, list):
            return await self.fetch_multiple_bodies([(u, headers) for u in url], research_goal)

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
                if self.return_summary and self.openai_client:
                    body = await self._make_summary(body, research_goal=research_goal)
                    if self.debug:
                        logger.info(f"Summary (url={url}, research_goal={research_goal}):\n{body}")

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


    async def fetch_multiple_bodies(self, requests: List[Tuple[str, Optional[Dict[str, str]]]], research_goal: str = None) -> List[dict]:
        async def fetch(url, headers, research_goal):
            return await self.fetch_body(url, headers, research_goal)

        tasks = [fetch(url, headers, research_goal) for url, headers in requests]
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
        openai_api_key: str = None,
        openai_base_url: str = None,
        openai_model: str = "gpt-4.1",
        return_summary: bool = False,
        summary_system_prompt: str = None,
        name=None,
        spec=None,
        instruction = None,
        is_dynamic = False,
        debug: bool = False
    ):
        self.web_scraper = WebScraper(
            default_user_agent=default_user_agent,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
            return_summary=return_summary,
            summary_system_prompt=summary_system_prompt,
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
                                "description": "List of URLs to scrape."
                            },
                            "research_goal": {
                                "type": "string",
                                "description": "The user's goal or question to investigate based on the given URLs."
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            self.web_scraper.fetch_body,
            instruction,
            is_dynamic
        )
