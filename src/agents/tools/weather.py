import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")


def weather_tool(city: str) -> str:
    if not API_KEY:
        return "Clé API météo manquante. Vérifiez votre fichier .env."

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": f"{city},FR",
            "appid": API_KEY,
            "units": "metric",
            "lang": "fr"
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            return "Impossible de récupérer la météo actuellement."

        city_name = data.get("name", city)
        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        description = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]

        return (
            f"Météo à {city_name} : {description}, {temperature:.1f}°C "
            f"(ressenti {feels_like:.1f}°C), humidité {humidity}%."
        )

    except requests.exceptions.RequestException:
        return "Erreur réseau lors de la récupération de la météo."
    except Exception:
        return "Une erreur inattendue est survenue."