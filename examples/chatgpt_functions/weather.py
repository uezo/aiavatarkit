import aiohttp
from datetime import datetime, timezone, timedelta

class WeatherFunction:
    def __init__(self, google_api_key: str, weather_appid: str):
        self.google_api_key = google_api_key
        self.weather_appid = weather_appid
        self.name = "get_weather"
        self.description="Get the weather forecast in a given location"
        self.parameters = {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string"
                }
            }
        }

    async def get_weather(self, location, tz_offset=9) -> dict:
        ret = {
            "description": "Compose a summary of the `weather_forecasts` information as a response message.",
            "weather_forecasts": []
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={self.google_api_key}"
                ) as resp:
                geo = await resp.json()

            lat = geo["results"][0]["geometry"]["location"]["lat"]
            lon = geo["results"][0]["geometry"]["location"]["lng"]

            async with session.get(
                    f"http://api.openweathermap.org/data/2.5/forecast?APPID={self.weather_appid}&lat={lat}&lon={lon}&cnt=9"
                ) as resp:
                weather = await resp.json()


        for v in weather["list"]:
            w = {}

            w["time"] = datetime.fromtimestamp(v["dt"], timezone(timedelta(hours=tz_offset))).isoformat()
            if v["weather"][0]["main"] == "Clear":
                w["weather"] = "clear"
            elif v["weather"][0]["main"] == "Clouds":
                w["weather"] = "clouds"
            elif v["weather"][0]["main"] == "Rain":
                w["weather"] = "rain"
            elif v["weather"][0]["main"] == "Snow":
                w["weather"] = "snow"
            w["temperature"] = str(int(v["main"]["temp"] - 273.15))

            ret["weather_forecasts"].append(w)

        return ret
