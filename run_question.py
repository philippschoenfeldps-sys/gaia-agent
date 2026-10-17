"""
Run the Agent on One Cached Question

Usage (from the project root, venv active):
    python run_question.py      -> lists all questions with their index
    python run_question.py 0    -> runs the agent on question 0

Why one question at a time?
Agents are slow and non-deterministic. 
Testing a single question and reading its full trace (Thought -> Code -> Observation) teaches far more than a full run.
"""

import json
import sys
from pathlib import Path

QUESTIONS_PATH = Path("data") / "questions.json"

def load_questions() -> list[dict]:
    """Load the questions cached by fetch_questions.py."""
    if not QUESTIONS_PATH.exists():

        # Error message
        sys.exit("data/questions.json not found -> run 'python fetch_questions.py' first.")

    with QUESTIONS_PATH.open(encoding = "utf-8") as f:
        return json.load(f)

def list_questions(questions: list[dict]) -> None:
    """Print a short overview so you can pick a question index."""
    for i, q in enumerate(questions):

        # Mark questions with attachments (those need file tools)
        attachment = f"  [file: {q['file_name']}]" if q.get("file_name") else ""

        # Shorten long questions to keep the list readable
        preview = q["question"][:90].replace("\n", " ")

        print(f"{i:>2}: {preview}...{attachment}")

def main() -> None:
    questions = load_questions()

    # sys.argv holds the command-line arguments: ["run_question.py", "0"]
    if len(sys.argv) < 2:
        list_questions(questions)
        print("\nRun one with: python run_question.py <index>")
        return

    index = int(sys.argv[1])
    q = questions[index]

    print("=" * 80)
    print(f"Question {index} (task_id {q['task_id']}):\n{q['question']}")
    print("=" * 80)

    # Import here (not at the top) so that just listing questions stays fast: building the agent loads smolagents, LiteLLM and connects to the model.
    from agent import GaiaAgent

    agent = GaiaAgent()
    answer = agent(q["question"], q["task_id"], q.get("file_name", ""))     # Agent can find attachments.

    print("\n" + "=" * 80)
    print(f"FINAL ANSWER: {answer!r}")      # Show quotes
    print("=" * 80)
 
 
if __name__ == "__main__":
    main()
