from fastmcp import FastMCP

mcp = FastMCP("Weather Info")

@mcp.tool(name="get_weather", description="Get weather information in the given location.")
async def get_weather_info(location: str):
    return {"location": location, "weahter": "clear", "temperature": 32.1}

@mcp.tool(name="get_alert", description="Get alert information in the given location")
async def get_alert_info(location: str):
    return {"location": location, "alert": "Thunderstorm"}

if __name__ == "__main__":
    mcp.run(transport="stdio")
