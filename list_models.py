"""
List Gemini Models

Prints the exact Gemini Model IDs the API can use.
The IDs go into MODEL_ID / FALLBACK_MODELS in .env, prefixed with "gemini/".

Run: python list_models.py
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()           # Reads GEMINI_API_KEY from .env
key = os.getenv("GEMINI_API_KEY")

if not key:
    raise SystemExit("GEMINI_API_KEY is missing in .env")

# The "models" endpoint lists every model available to this key.
r = requests.get(
    "https://generativelanguage.googleapis.com/v1beta/models",
    params = {"key": key, "pageSize": 200},
    timeout = 30,
)
r.raise_for_status()
 
for m in r.json().get("models", []):
    # Only models that can generate text are useful for the agent.
    if "generateContent" in m.get("supportedGenerationMethods", []):
        # m["name"] looks like "models/gemini-...-flash" -> strip the "models/" prefix.
        print(f'gemini/{m["name"].removeprefix("models/"):<45} | {m.get("displayName", "")}')
