from logging import getLogger
from typing import Callable
import httpx
from aiavatar.sts.llm import Tool

logger = getLogger(__name__)


class GrokSearch:
    def __init__(self,
        *,
        xai_api_key: str,
        model: str = "grok-4-fast-non-reasoning-latest",
        system_prompt: str = None,
        temperature: float = 0.0,
        language: str = None,
        make_query: Callable[[str], str] = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 60.0,
        debug: bool = False
    ):
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )
        self.xai_api_key = xai_api_key
        self.model = model
        self.system_prompt = system_prompt or "Search the web to answer the user's query. Base your response strictly on the search results, and do not include your own opinions."
        self.temperature = temperature
        self.language = language
        self.make_query = make_query
        self.debug = debug

    async def search(self, query: str, sources: list = None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.xai_api_key}"
        }

        if self.make_query:
            query = self.make_query(query)

        if self.debug:
            logger.info(f"Grok Search Query: {query}")

        payload = {
            "messages": [
                {"role": "system", "content": self.system_prompt+ f"\nOutput language code: {self.language}" if self.language else ""},
                {"role": "user", "content": f"Search: {query}"}
            ],
            "search_parameters": {
                "mode": "auto",
                "sources": sources or [{"type": "web"}]
            },
            "model": self.model
        }

        resp = await self.http_client.post(
            url="https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if resp.status_code != 200:
            logger.error(f"Error at Grok web search tool: {resp.read()}")
            return {"error": f"Error at Grok web search tool"}

        search_result = resp.json()["choices"][0]["message"]["content"]
        if self.debug:
            logger.info(f"Grok Search Result: {search_result}")

        return {"search_result": search_result}


class GrokSearchTool(Tool):
    def __init__(
        self,
        *,
        xai_api_key: str,
        model: str = "grok-4-fast-non-reasoning-latest",
        system_prompt: str = None,
        temperature: float = 0.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: int = 30000,
        name=None,
        spec=None,
        instruction = None,
        is_dynamic = False,
        debug: bool = False
    ):
        self.grok_web_search = GrokSearch(
            xai_api_key=xai_api_key,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            debug=debug
        )
        super().__init__(
            name or "grok_web_search",
            spec or {
                "type": "function",
                "function": {
                    "name": name or "grok_web_search",
                    "description": "Search the web using Grok WebSearch",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    },
                }
            },
            self.grok_web_search.search,
            instruction,
            is_dynamic
        )
