from logging import getLogger
from typing import Callable
from urllib.parse import urlparse, parse_qs
import openai
from aiavatar.sts.llm import Tool

logger = getLogger(__name__)


class OpenAIWebSearch:
    def __init__(self,
        *,
        openai_api_key: str,
        system_prompt: str = None,
        base_url: str = None,
        model: str = "gpt-5-search-api",
        temperature: float = 0.5,
        search_context_size: str = "medium",
        country: str = None,
        language: str = None,
        make_query: Callable[[str], str] = None,
        timeout: int = 30000,
        debug: bool = False
    ):
        if "azure" in model:
            api_version = parse_qs(urlparse(base_url).query).get("api-version", [None])[0]
            self.openai_client = openai.AsyncAzureOpenAI(
                api_key=openai_api_key,
                api_version=api_version,
                base_url=base_url,
                timeout=timeout
            )
        else:
            self.openai_client = openai.AsyncClient(api_key=openai_api_key, base_url=base_url, timeout=timeout)

        self.system_prompt = system_prompt or "Search the web to answer the user's query. Base your response strictly on the search results, and do not include your own opinions."
        self.model = model
        self.temperature = temperature
        self.search_context_size = search_context_size
        self.country = country
        self.language = language
        self.make_query = make_query
        self.debug = debug

    async def search(self, query: str):
        web_search_options = {
            "search_context_size": self.search_context_size
        }
        if self.country:
            web_search_options["user_location"] = {
                "type": "approximate",
                "approximate": {
                    "country": self.country
                }
            }

        if self.make_query:
            query = self.make_query(query)

        if self.debug:
            logger.info(f"OpenAI WebSearch Query: {query}")

        response = await self.openai_client.chat.completions.create(
            model=self.model,
            web_search_options=web_search_options,
            messages=[
                {"role": "system", "content": self.system_prompt + f"\nOutput language code: {self.language}" if self.language else ""},
                {"role": "user", "content": f"Search: {query}"}
            ],
        )

        search_result = response.choices[0].message.content
        if self.debug:
            logger.info(f"OpenAI WebSearch Result: {search_result}")

        return {"search_result": search_result}


class OpenAIWebSearchTool(Tool):
    def __init__(
        self,
        *,
        openai_api_key: str,
        system_prompt: str = None,
        base_url: str = None,
        model: str = "gpt-4o-search-preview",
        temperature: float = 0.5,
        search_context_size: str = "medium",
        country: str = "JP",
        language: str = None,
        make_query: Callable[[str], str] = None,
        timeout: int = 30000,
        name=None,
        spec=None,
        instruction = None,
        is_dynamic = False,
        debug: bool = False
    ):
        self.openai_web_search = OpenAIWebSearch(
            openai_api_key=openai_api_key,
            system_prompt=system_prompt,
            base_url=base_url,
            model=model,
            temperature=temperature,
            search_context_size=search_context_size,
            country=country,
            language=language,
            make_query = make_query,
            timeout=timeout,
            debug=debug
        )
        super().__init__(
            name or "web_search",
            spec or {
                "type": "function",
                "function": {
                    "name": name or "web_search",
                    "description": "Search the web using OpenAI WebSearch",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    },
                }
            },
            self.openai_web_search.search,
            instruction,
            is_dynamic
        )
