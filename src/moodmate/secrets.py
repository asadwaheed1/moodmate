import os
from dotenv import load_dotenv
from rich import print

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_api_url = os.getenv("GEMINI_API_URL")
gemini_api_model = os.getenv("GEMINI_API_MODEL")

if not gemini_api_key or not gemini_api_url or not gemini_api_model:
    print("[red]Please set the GEMINI_API_KEY, GEMINI_API_URL, and GEMINI_API_MODEL environment variables.[/red]")
    exit(1)

secrets = {
    "GEMINI_API_KEY": gemini_api_key,
    "GEMINI_API_URL": gemini_api_url,
    "GEMINI_API_MODEL": gemini_api_model,
}


