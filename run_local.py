"""
Run the Agent on All Questions (with an answer cache)

Usage (from the project root, venv active):
    python run_local.py             -> answer every question that has no cached answer yet
    python run_local.py --retry     -> also re-run questions whose cached answer is empty
    python run_local.py --only 3 7  -> run only the questions with index 3 and 7 (overwrites them)
    python run_local.py --show      -> just print the current answer table, run nothing
"""

import argparse
import json
import time
from pathlib import Path

DATA_DIR = Path("data")
QUESTIONS_PATH = DATA_DIR / "questions.json"
ANSWERS_PATH = DATA_DIR / "answers.json"
 
# Pause between questions in seconds. DuckDuckGo and free LLM APIs rate-limit bursts of requests.
# A short pause avoids "429 Too Many Requests" errors.
PAUSE_SECONDS = 5
 
def load_json(path: Path, default):
    """Read a JSON file, or return `default` if it does not exist yet."""
    if not path.exists():
        return default
    
    with path.open(encoding="utf-8") as f:
        return json.load(f)
 
def save_answers(answers: dict) -> None:
    """Write the answer cache. Called after EVERY question -> crash-safe."""
    with ANSWERS_PATH.open("w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)
 
def print_table(questions: list[dict], answers: dict) -> None:
    """Overview of all questions and their cached answers."""
    print(f"\n{'#':>2}  {'file':<6} {'answer':<40} question")
    print("-" * 100)
    for i, q in enumerate(questions):
        entry = answers.get(q["task_id"], {})
        answer = entry.get("answer", "—")  # "—" = not run yet
        if answer == "":
            answer = "(empty)"
        file_type = Path(q["file_name"]).suffix if q.get("file_name") else ""
        print(f"{i:>2}  {file_type:<6} {answer[:40]:<40} {q['question'][:45].replace(chr(10), ' ')}")
    done = sum(1 for e in answers.values() if e.get("answer"))
    print(f"\nAnswered (non-empty): {done}/{len(questions)}")
 
def main() -> None:
    # argparse turns command-line flags into Python variables (standard library).
    parser = argparse.ArgumentParser(description = "Run the GAIA agent locally with caching.")
    parser.add_argument("--retry", action = "store_true", help = "re-run questions with empty answers")
    parser.add_argument("--only", type = int, nargs = "+", help = "run only these question indices")
    parser.add_argument("--show", action = "store_true", help = "only print the answer table")
    args = parser.parse_args()
 
    questions = load_json(QUESTIONS_PATH, None)
    if questions is None:
        raise SystemExit("data/questions.json not found -> run 'python fetch_questions.py' first.")
 
    # The cache maps task_id -> {"answer": ..., "seconds": ...}.
    # Keyed by task_id (not by index), so it stays correct even if the question order changes.
    answers: dict = load_json(ANSWERS_PATH, {})
 
    if args.show:
        print_table(questions, answers)
        return
 
    # Decide which questions to run
    if args.only:
        todo = [(i, questions[i]) for i in args.only]
    else:
        todo = []

        for i, q in enumerate(questions):
            cached = answers.get(q["task_id"])

            if cached is None:
                todo.append((i, q))                      # Never run
            elif args.retry and not cached.get("answer"):
                todo.append((i, q))                      # Ran, but produced nothing
    if not todo:
        print("Nothing to do: all questions have cached answers. Use --retry or --only.")
        print_table(questions, answers)
        return
 
    print(f"Running {len(todo)} question(s)...")
 
    # Build the agent once and reuse it: creating it loads the model connection and tools.
    from agent import GaiaAgent
    agent = GaiaAgent()
 
    for n, (i, q) in enumerate(todo, start=1):
        print("\n" + "#" * 100)
        print(f"[{n}/{len(todo)}] Question {i}: {q['question'][:150]}")
        print("#" * 100)
 
        start = time.perf_counter()     # precise timer for measuring durations
        try:
            answer = agent(q["question"], q["task_id"], q.get("file_name", ""))
        except KeyboardInterrupt:

            # Ctrl+C: stop cleanly; everything answered so far is already saved.
            print("\nStopped by user. Progress is saved.")
            break
        seconds = round(time.perf_counter() - start, 1)
 
        answers[q["task_id"]] = {"answer": answer, "seconds": seconds, "index": i}
        save_answers(answers)           # save immediately (crash-safe)
        print(f"\n-> Answer: {answer!r}  ({seconds}s)")
 
        if n < len(todo):
            time.sleep(PAUSE_SECONDS)
 
    print_table(questions, answers)
 
 
if __name__ == "__main__":
    main()

