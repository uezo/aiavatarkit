from logging import getLogger
from typing import Callable
from google import genai    # pip install google-genai
from google.genai import types
from aiavatar.sts.llm import Tool

logger = getLogger(__name__)


class GeminiWebSearch:
    def __init__(self,
        *,
        gemini_api_key: str,
        model: str = "gemini-2.5-flash",
        system_prompt: str = None,
        temperature: float = 0.0,
        thinking_budget: int = -1,
        language: str = None,
        make_query: Callable[[str], str] = None,
        yield_start_message: str = None,
        timeout: int = 30000,
        debug: bool = False
    ):
        self.gemini_client = genai.Client(
            api_key=gemini_api_key,
            http_options=types.HttpOptions(timeout=timeout)
        )
        self.model = model
        self.system_prompt = system_prompt or "Search the web to answer the user's query. Base your response strictly on the search results, and do not include your own opinions."
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.language = language
        self.make_query = make_query
        self.yield_start_message = yield_start_message
        self.debug = debug
        self.search_tool = types.Tool(google_search=types.GoogleSearch())

    async def search(self, query: str):
        thinking_config = None
        if self.thinking_budget >= 0:
            thinking_config = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )

        if self.make_query:
            query = self.make_query(query)

        if self.debug:
            logger.info(f"Gemini Search Query: {query}")

        if self.yield_start_message:
            yield {"progress": "start"}
            yield self.yield_start_message

        resp = await self.gemini_client.aio.models.generate_content(
            model=self.model,
            config = types.GenerateContentConfig(
                system_instruction=self.system_prompt + f"\nOutput language code: {self.language}" if self.language else "",
                temperature=self.temperature,
                tools=[self.search_tool],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=thinking_config
            ),
            contents=f"Search: {query}",
        )

        search_result = resp.candidates[0].content.parts[0].text

        if self.debug:
            logger.info(f"Gemini Search Result: {search_result}")

        yield {"search_result": search_result}, True


class GeminiWebSearchTool(Tool):
    def __init__(
        self,
        *,
        gemini_api_key: str,
        model: str = "gemini-2.5-flash",
        system_prompt: str = None,
        temperature: float = 0.0,
        thinking_budget: int = -1,
        language: str = None,
        make_query: Callable[[str], str] = None,
        timeout: int = 30000,
        name=None,
        spec=None,
        instruction = None,
        is_dynamic = False,
        debug: bool = False
    ):
        self.gemini_web_search = GeminiWebSearch(
            gemini_api_key=gemini_api_key,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            thinking_budget=thinking_budget,
            language=language,
            make_query=make_query,
            timeout=timeout,
            debug=debug
        )
        super().__init__(
            name or "google_search",
            spec or {
                "type": "function",
                "function": {
                    "name": name or "google_search",
                    "description": "Search the web using Google",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    },
                }
            },
            self.gemini_web_search.search,
            instruction,
            is_dynamic
        )
