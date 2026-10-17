"""
test_tools.py  –  Test level 2: do the tools work ON THEIR OWN?

Calls every tool directly, WITHOUT the LLM → costs zero API quota.
If a tool is broken here, the agent can never succeed with it, no matter how good the model is.
That's why we test tools first: it separates "tool bug" from "model mistake".

Run:  python test_tools.py            → all tools
      python test_tools.py wiki audio → only selected groups (wiki, files, audio, youtube)
"""

import json
import re
import sys
from pathlib import Path

import tools

FILES_DIR = Path("data") / "files"
questions = json.loads((Path("data") / "questions.json").read_text(encoding="utf-8"))


def show(title: str, result: str, max_len: int = 600) -> None:
    """Print a tool result in a compact, readable way."""
    print(f"\n--- {title} ---")
    text = str(result)
    print(text[:max_len] + (" […]" if len(text) > max_len else ""))
    # Our tools return errors as text (summary 4.4) → detect them here.
    if text.lower().startswith(("error", "could not")):
        print(">>> ⚠ the tool reported an error")


def file_with_suffix(suffix: str) -> Path | None:
    """Find the attached file with a given extension, e.g. '.xlsx'."""
    for q in questions:
        if q.get("file_name", "").endswith(suffix):
            return FILES_DIR / q["file_name"]
    return None


# Which test groups to run: all by default, or those named on the command line.
groups = set(sys.argv[1:]) or {"wiki", "files", "audio", "youtube"}

if "wiki" in groups:
    show("wikipedia_search", tools.wikipedia_search("Mercedes Sosa"))
    show("wikipedia_page (section)", tools.wikipedia_page("Mercedes Sosa", section="Discography"))
    # Old revision: should show the article as it was at the end of 2022.
    show("wikipedia_page (as_of_date)", tools.wikipedia_page("Mercedes Sosa", section="Discography",
                                                             as_of_date="2022-12-31"))
    # Error case: the tool must return a readable message, not crash.
    show("wikipedia_page (missing section)", tools.wikipedia_page("Mercedes Sosa", section="xyz"))

if "files" in groups:
    xlsx = file_with_suffix(".xlsx")
    show(f"read_spreadsheet({xlsx.name if xlsx else None})",
         tools.read_spreadsheet(str(xlsx)) if xlsx else "no .xlsx found")
    py = file_with_suffix(".py")
    show(f"read_text_file({py.name if py else None})", tools.read_text_file(str(py)) if py else "no .py found")
    show(f"run_python_file({py.name if py else None})", tools.run_python_file(str(py)) if py else "no .py found")

if "audio" in groups:
    # First call downloads the Whisper model once (a few hundred MB) → can take a while.
    mp3 = file_with_suffix(".mp3")
    show(f"transcribe_audio({mp3.name if mp3 else None})",
         tools.transcribe_audio(str(mp3)) if mp3 else "no .mp3 found")

if "youtube" in groups:
    # Test EVERY YouTube link that appears in the question texts.
    # Note: a video without subtitles returns an error by design (e.g. a question about what is SHOWN).
    urls = sorted({m.group(0) for q in questions
                   for m in re.finditer(r"https?://(?:www\.)?youtube\.com/watch\?v=[\w-]{11}", q["question"])})
    for url in urls:
        show(f"youtube_transcript({url})", tools.youtube_transcript(url))
    if not urls:
        print("\n--- youtube_transcript ---\nno YouTube link found")
        
print("\nDone. Check each block: real content = tool works; an error text = fix the tool first.")