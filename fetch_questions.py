"""
Local Question Cache

What this script does:
1. Downloads all evaluation questinos from the course API once.
2. Saves them to data/questions.json (local cache)
3. Downloads every attached file (images, audio, Excel, Python) into data/files/.

Why a cache?
- The agent is tested many times: re-downloading every time is slow and unnecessary.
- Working on a fixed local copy makes runs reproducible.
- It also works offline.

Run it from the project root with: python fetch_questions.py.
"""

import json
from collections import Counter     # Convenient way to count things (e.g., file types).
from pathlib import Path            # For working with files and directory paths

import requests                     # For sending HTTP requests to web server/APIs
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://agents-course-unit4-scoring.hf.space"        # Course API
DATA_DIR = Path("data")                                         # Folder for all local data
FILES_DIR = DATA_DIR / "files"  
QUESTIONS_PATH = DATA_DIR / "questions.json"                    # Where the questions are cached
TIMEOUT = 30                                                    # Seconds before a requests gives up

def fetch_questions() -> list[dict]:
    """Fetch all questions from the API and return them as list of dicts."""
    url = f"{API_URL}/questions"

    # Always need a timeout: without it, a hanging server would block the script forever (request would wait forever).
    response = requests.get(url, timeout = TIMEOUT)

    # Raise an HTTPError for 4xx/5xx status code (4... = client error & 5... server error)
    response.raise_for_status()

    # The API returns JSON: .json() parses it into Python objects (here a list of dicts).
    questions = response.json()
    print(f"Fetched {len(questions)} questions from the API.")

    return questions

def save_questions(questions: list[dict]) -> None:
    """Write the questions to QUESTION_PATH as nicely formatted JSON."""
    with QUESTIONS_PATH.open("w", encoding = "utf-8") as f:         # Encoding avoids problems with special characters
        
        # intend for human-readable formatting; ensure_ascii to keep characters like "é" or "ł" readable instead of "é"
        json.dump(questions, f, indent = 2, ensure_ascii = False)

    print(f"Saved questions to {QUESTIONS_PATH}")

# Fallback source: the course questions come from the GAIA validation set on the Hugging Face Hub (the course API's / files endpoint return 404).
GAIA_REPO = "gaia-benchmark/GAIA"
GAIA_FOLDER = "2023/validation"

def download_from_gaia_dataset(file_name: str, target: Path) -> Path | None:
    """Fallback: download one attachment from the gated GAIA dataset on the Hugging Face Hub (never the metadata files)."""
    try:
        from huggingface_hub import hf_hub_download

        # Store the file in the HF cache and return that local path.
        cached_path = hf_hub_download(
            repo_id = GAIA_REPO,
            repo_type = "dataset",                      # Dataset repo (not a model repo)
            filename = f"{GAIA_FOLDER}/{file_name}",
        )
    except Exception as e:

        # Typical causes: conditions not accepted yet (403 / GatedRepoError) or not logged in.
        print(f"  [error] GAIA dataset fallback failed for {file_name}: {e}")
        return None

    # Copy from the HF cache into own data/files folder.
    target.write_bytes(Path(cached_path).read_bytes())
    size_kb = target.stat().st_size / 1024
    print(f"  [ok]  {file_name} ({size_kb:.1f} KB, from GAIA dataset)")

    return target


def download_file(task_id: str, file_name: str) -> Path | None:
    """Download the attachement of a task into FILES_DIR and return its path or None if the download failed."""
    target = FILES_DIR / file_name

    # Cache check: if the file is already there, skip the network request entirely.
    if target.exists():
        print(f"  [skip] {file_name} already exists")

        return target

    url = f"{API_URL}/files/{task_id}"

    try:
        response = requests.get(url, timeout = TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:

        # Base class for all requests errors (timeout, connection errors, HTTP errors ...).
        print(f"  [warn] course API failed for {file_name} ({e}) -> trying GAIA dataset")
        
        return download_from_gaia_dataset(file_name, target)

    # Files can be binary (images, audio, Excel): write raw bytes, not text.
    target.write_bytes(response.content)        # Opens, writes and closes the file.
    size_kb = len(response.content) / 1024
    print(f"  [ok]  {file_name} ({size_kb:.1f} KB)")

    return target

def print_summary(questions: list[dict]) -> None:
    """Small data analysis: how many questions have which type of attachment?"""
    file_types = Counter(

        # Extract the file extension (e.g, mp3); questions without a file get the label "(no file)".
        Path(q["file_name"]).suffix if q.get("file_name") else "(no file)"      
        for q in questions
    )
    levels = Counter(q.get("Level") for q in questions)

    print("\n--- Summary ---")
    print(f"Questions: {len(questions)}")
    print(f"Levels:    {dict(levels)}")
    print("Attachments by type:")
    
    for file_type, count in file_types.most_common():       # Sorts by count, highest first.
        print(f"  {file_type:<10} {count}")

def main() -> None:

    # Also create "data/" if it does not exist yet; no error if the folder already exists.
    FILES_DIR.mkdir(parents = True, exist_ok = True)   

    questions = fetch_questions()
    save_questions(questions)

    print("\nDownloading attachements...")

    for q in questions:
        if q.get("file_name"):      # Is only True for questions that really have an attachment.
            download_file(q["task_id"], q["file_name"])

    print_summary(questions)

# Runs when the file is executed directly (python fetch_questions.py).
if __name__ == "__main__":
    main()