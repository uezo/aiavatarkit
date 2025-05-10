from logging import getLogger
import httpx
from aiavatar.sts.llm import Tool

logger = getLogger(__name__)


class BingSearch:
    def __init__(self,
        *,
        subscription_key: str,
        endpoint: str = "https://api.bing.microsoft.com/v7.0/search",
        mkt: str = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        debug: bool = False
    ):
        self.subscription_key = subscription_key
        self.endpoint = endpoint
        self.mkt = mkt
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections
            )
        )
        self.debug = debug

    async def search(self, query: str, count: int = 5):
        params = {"q": query, "count": count}
        if self.mkt:
            params["mkt"] = self.mkt

        if self.debug:
            logger.info(f"Search: {params}")

        response = await self.http_client.get(
            url=self.endpoint,
            headers={"Ocp-Apim-Subscription-Key": self.subscription_key},
            params=params
        )
        response.raise_for_status()
        return response.json().get("webPages", {}).get("value", [])


class BingSearchTool(Tool):
    def __init__(
        self,
        *,
        subscription_key: str,
        endpoint: str = "https://api.bing.microsoft.com/v7.0/search",
        mkt: str = None,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 10.0,
        name=None,
        spec=None,
        instruction = None,
        is_dynamic = False,
        debug: bool = False
    ):
        self.bing_search = BingSearch(
            subscription_key=subscription_key,
            endpoint=endpoint,
            mkt=mkt,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            timeout=timeout,
            debug=debug
        )
        super().__init__(
            name or "bing_search",
            spec or {
                "type": "function",
                "function": {
                    "name": name or "bing_search",
                    "description": "Search the web using Bing",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        }
                    },
                }
            },
            self.bing_search.search,
            instruction,
            is_dynamic
        )
