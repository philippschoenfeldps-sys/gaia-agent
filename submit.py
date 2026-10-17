"""
Submit the Cached Answer to the Course Leaderboard

Usage (from the project root, venv active):
    python submit.py            -> shows what would be sent (dry run, nothing is submitted)
    python submit.py --send     -> really submits
"""

import argparse
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from agent import ANSWER_RULES

load_dotenv()               # Reads HF_USERNAME and CODE_URL from .env

API_URL = "https://agents-course-unit4-scoring.hf.space"
ANSWERS_PATH = Path("data") / "answers.json"

def main() -> None:
    parser = argparse.ArgumentParser(description="Submit cached GAIA answers.")
    parser.add_argument("--send", action = "store_true", help = "really submit (default: dry run)")
    args = parser.parse_args()
 
    username = os.getenv("HF_USERNAME", "").strip()
    code_url = os.getenv("CODE_URL", "").strip()
    
    if not username or not code_url:
        raise SystemExit("Set HF_USERNAME and CODE_URL in your .env file first.")
 
    with ANSWERS_PATH.open(encoding="utf-8") as f:
        cache = json.load(f)
 
    # Convert the cache format {task_id: {"answer": ...}} into the API format.
    # Empty answers are skipped.
    answers = [
        {"task_id": task_id, "submitted_answer": entry["answer"]}
        for task_id, entry in cache.items()
        if entry.get("answer")
    ]
 
    payload = {"username": username, "agent_code": code_url, "answers": answers}
 
    print(f"Username:   {username}")
    print(f"Code link:  {code_url}")
    print(f"Answers:    {len(answers)} (of {len(cache)} cached)")
    for a in answers:
        print(f"  {a['task_id'][:8]}…  {a['submitted_answer']!r}")
 
    if not args.send:
        print("\nDry run only. Add --send to submit for real.")
        return
 
    # Send the payload as JSON in the request body.
    response = requests.post(f"{API_URL}/submit", json=payload, timeout=60)

    if response.status_code != 200:

        # Show the server's error message (e.g. validation problems).
        raise SystemExit(f"Submission failed ({response.status_code}): {response.text}")
 
    result = response.json()
    print("\n=== Result ===")
    print(f"Score:   {result['score']}%  ({result['correct_count']}/{result['total_attempted']} correct)")
    print(f"Message: {result['message']}")
 
 
if __name__ == "__main__":
    main()
