from aiavatar.sts.llm.base import Tool


def make_tool(func, **kwargs):
    return Tool(
        name="test_tool",
        spec={
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "test",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }
        },
        func=func,
        **kwargs
    )


def test_response_formatter_decorator():
    async def my_func(query: str):
        return {"answer": query}

    tool = make_tool(my_func)
    assert tool._response_formatter is None

    @tool.response_formatter
    def format_response(result, arguments):
        return f"Query: {arguments['query']} / Answer: {result['answer']}"

    assert tool._response_formatter is not None
    assert tool._response_formatter({"answer": "hello"}, {"query": "hi"}) == "Query: hi / Answer: hello"


def test_response_formatter_not_set_by_default():
    async def my_func(query: str):
        return {"answer": query}

    tool = make_tool(my_func)
    assert tool._response_formatter is None
