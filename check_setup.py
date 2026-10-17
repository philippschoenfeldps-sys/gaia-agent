"""
Is the environment configured correctly?

Checks (in this order, cheapest first):
1. .env is found and the important values are set
2. all required packages can be imported
3. the cached questions and attachments exist
4. the LLM answers a tiny prompt  (costs exactly ONE request of your daily quota)

Run: python check_setup.py
Run without the LLM call: python check_setup.py --no-llm
"""

import importlib
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"
problems = 0                        # Counts failed checks -> exit code at the end


def check(condition: bool, ok_msg: str, fail_msg: str, warn_only: bool = False) -> None:
    """Print one result line and count failures."""
    global problems

    if condition:
        print(f"{OK} {ok_msg}")
    elif warn_only:
        print(f"{WARN} {fail_msg}")
    else:
        print(f"{FAIL} {fail_msg}")
        problems += 1


print("=== 1. Configuration (.env) ===")
env_path = find_dotenv(usecwd=True)             # Searches the current folder (and parents) for .env
check(bool(env_path), f".env found: {env_path}", ".env not found → create it in the project root")
load_dotenv(env_path)

model_id = os.getenv("MODEL_ID", "")
check(bool(model_id), f"MODEL_ID = {model_id}", "MODEL_ID is not set")

# Placeholders like "<id-...>" mean the real ID was not filled in yet.
check("<" not in model_id, "MODEL_ID contains no placeholder", "MODEL_ID still contains a <placeholder>")

if model_id.startswith("gemini/"):
    key = os.getenv("GEMINI_API_KEY", "")

    # Never print the full key: only the first characters, to recognize it.
    check(len(key) > 20 and "your_" not in key,
          f"GEMINI_API_KEY set ({key[:6]}…, {len(key)} characters)",
          "GEMINI_API_KEY missing or still the placeholder")
    fallbacks = os.getenv("FALLBACK_MODELS", "")
    check("<" not in fallbacks, f"FALLBACK_MODELS = {fallbacks or '(none)'}",
          "FALLBACK_MODELS still contains a <placeholder>")

check(bool(os.getenv("HF_USERNAME")) and "your_" not in os.getenv("HF_USERNAME", ""),
      f"HF_USERNAME = {os.getenv('HF_USERNAME')}",
      "HF_USERNAME not set (only needed for submit.py)", warn_only=True)

print("\n=== 2. Packages ===")
# import name -> pip package name
packages = {
    "smolagents": "smolagents[litellm]", "litellm": "smolagents[litellm]",
    "ddgs": "ddgs", "markdownify": "markdownify", "pandas": "pandas", "openpyxl": "openpyxl",
    "faster_whisper": "faster-whisper", "youtube_transcript_api": "youtube-transcript-api",
    "dotenv": "python-dotenv", "requests": "requests",
}
for module, pip_name in packages.items():
    try:
        importlib.import_module(module)
        check(True, f"{module}", "")
    except ImportError:
        check(False, "", f"{module} missing → python -m pip install \"{pip_name}\"")

# Importing tools.py also validates every @tool docstring (catches the "Args: " bug).
try:
    import tools            # noqa: F401
    check(True, "tools.py imports (all tool docstrings valid)", "")
except Exception as e:
    check(False, "", f"tools.py fails to import: {e}")

print("\n=== 3. Data ===")
q_path = Path("data") / "questions.json"
check(q_path.exists(), "data/questions.json exists", "data/questions.json missing → python fetch_questions.py")
files = list((Path("data") / "files").glob("*"))
check(len(files) >= 5, f"{len(files)} attachments in data/files", f"only {len(files)} attachments → python fetch_questions.py")

print("\n=== 4. LLM connection ===")
if "--no-llm" in sys.argv:
    print("skipped (--no-llm)")
else:
    try:
        from agent import build_model

        model = build_model()

        # smolagents models take a list of chat messages
        reply = model([{"role": "user", "content": [{"type": "text", "text": "Reply with exactly: OK"}]}])
        check(True, f"model answered: {reply.content.strip()[:50]!r}", "")
    except Exception as e:
        
        # Typical: wrong model ID (404), invalid key (400/403), quota exhausted (429).
        check(False, "", f"model call failed: {type(e).__name__}: {str(e)[:300]}")

print("\n" + ("All checks passed ✔" if problems == 0 else f"{problems} problem(s) found ✘"))
sys.exit(1 if problems else 0)