"""
GAIA Agent

Contains:
- build_model():        create the LLM backend from settings in the .env file
- normalize_answer():   cleans the final answer for exact-match scoring
- GaiaAgent:            wraps a smolagents CodeAgent

Design idea: local Ollama for testing, a free cloud API later for final test.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool

from fetch_questions import API_URL, FILES_DIR
from tools import FILE_TOOLS, KNOWLEDGE_TOOLS, fetch_page_markdown

load_dotenv()       # Get the key=value pairs

# Windows console fix: when output is piped (e.g., "| Tee-Object"), Python on Windows encodes text as cp1252, which cannot represent emojis.
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Silence harmless library noise: LiteLLM's background logging worker prints warnings about asyncio event loops on every call.
import logging
import warnings

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message="coroutine .* was never awaited")

MODEL_ID = os.getenv("MODEL_ID", "ollama_chat/qwen3.5:4b")              # LiteLLM model string
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:11434")              # Ollama's local server
NUM_CTX = int(os.getenv("NUM_CTX", "16384"))                            # Context window (tokens)
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))                           # Loop limit question
PLANNING_INTERVAL = int(os.getenv("PLANNING_INTERVAL", "4")) or None    # Every planning request is an extra LLM request.

# Answer rules: GAIA is scored by EXACT MATCH, so following rules are append to every question
ANSWER_RULES = """
When you have the answer, call final_answer() with only the answer itself:
- no explanation, no full sentence, no prefix like "FINAL ANSWER"
- numbers: digits only, no units and no thousands separators unless the question asks for them
- lists: comma-separated with a space after each comma, in the order the question requests
- follow every format instruction in the question exactly (plural, abbreviations, first name only, etc.)
- if the questions asks for a number with decimals (e.g., USD with two decimals), use exactly that format
"""

# Tool strategy hints: small models pick tools better when given the strategy explicitly.
TOOL_HINTS = """
Strategy:
- For facts that are on Wikipedia, prefer wikipedia_search + wikipedia_page over web search.
  Use the 'section' argument for long pages and 'as_of_date' if the question names a year/version.
- For YouTube links, use youtube_transcript.
- Use web_search + visit_webpage for everything else. Verify numbers on the source page.
- If a page is long or visit_webpage output is truncated, use search_webpage(url, query)
  to jump directly to the passages you need instead of visiting the page again.
- If you cannot access the needed information (e.g. a video without subtitles, an image you
  cannot see), say so with final_answer("") instead of guessing a plausible-looking answer.
- Never guess. If a tool returns an error, try a different query or tool.
- Work only with the question and its attachment. 
  Take attached data at face value: do not assume missing information (prices, other files, hidden sheets) unless the question mentions it.
- Once you have inspected the data, compute the result in the next step and call final_answer().
  Do not re-read the same data again.
"""

# A Pattern that detects garbage code:
# if the agent hits max_steps, smolagents forces a final answer from an unfinished state and weak models sometimes return their code as the "answer".
CODE_PATTERN = re.compile(r"`{2,}|\bimport \w+|final_answer\(|print\(")

class VisitWebpageWithHint(VisitWebpageTool):
    """
    Replacement for smolagents' visit_webpage (same name, same input, same description):
    - it downloads pages with a proper User-Agent.
    - if the page is cut off, the observation itself tells the agent to use search_webpage.
    """
    def forward(self, url: str) -> str:
        try:
            text = fetch_page_markdown(url)         # Shared helper from tools.py
        except Exception as e:
            return f"Error fetching the webpage: {e}"

        if len(text) <= self.max_output_length:
            return text

        return (
            text[: self.max_output_length]
            + "\n\n..._This content has been truncated_...\n"
            + "[HINT: Only the beginning of this page is shown. Do NOT visit it again. "
            + f'Call search_webpage(url="{url}", query="<2-4 key words>") to get the exact '
            + "passage you need.]"
        )

def build_model() -> LiteLLMModel:
    """Create the LLM backend. LiteLLM routes the request based on the MODEL_ID prefix."""
    kwargs = {"model_id": MODEL_ID}

    if MODEL_ID.startswith("ollama"):
        
        # Local Ollama needs the server address and a large enough context window. Agent prompts are long (system prompt + tool description + every step's observation).
        # If num_ctx is too small, Ollama silently cuts off the start of the prompt and the agent "forgets" its instructions.
        kwargs["api_base"] = API_BASE
        kwargs["num_ctx"] = NUM_CTX
    else:

        # Cloud free tiers answer "429 Too Many Requests" when limits are exceeded.
        kwargs["num_retries"] = int(os.getenv("NUM_RETRIES", "2"))

        if os.getenv("TEMPERATURE"):
            kwargs["temperature"] = float(os.getenv("TEMPERATURE"))
        
        # Free tiers count requests per model. If the main model's daily quota is used up, LiteLLM switches to the next model in this list (each has its own quota).
        fallbacks = [m.strip() for m in os.getenv("FALLBACK_MODELS", "").split(",") if m.strip()]
        
        if fallbacks:
            kwargs["fallbacks"] = fallbacks

        # smolagents' built-in rate limiter waits between requests so the per-minute limit is never hit (5 RPM = one request every 12 s).
        kwargs["requests_per_minute"] = float(os.getenv("REQUESTS_PER_MINUTE", "5"))

    return LiteLLMModel(**kwargs)

def normalize_answer(answer) -> str:
    """Clean up the agent's answer for exact-match scoring. The LLM often adds small extras that make a correct answer count as wrong."""
    text = str(answer).strip()

    # Remove prefixes like "FINAL ANSWER:", "Answer:" (case-insensitive, at the start only).
    text = re.sub(r"^(final\s+answer|answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)

    # Remove matching quotes or backticks around the whole answer: "Paris" -> Paris
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()

    # Remove one trailing period ("Paris." -> "Paris"), but keep decimals like "3.5" because those do not end with a period
    if text.endswith(".") and not text.endswith(".."):
        text = text[:-1]

    # Collapse whitespace/newlines into single spaces.
    text = re.sub(r"\s+", " ", text)

    # Normalize list seperators: "a,b ,c" -> "a, b, c"
    if "," in text:
        text = ", ".join(part.strip() for part in text.split(","))

    return text

def ensure_file(task_id: str, file_name: str) -> Path | None:
    """Make sure the attachment of a task exists locally."""
    path = FILES_DIR / file_name
    if path.exists():
        return path
    
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    from fetch_questions import download_file       # Reusing code.

    return download_file(task_id, file_name)

class GaiaAgent:
    """
    Callable wrapper around a CodeAgent.
    Usage: answer = agent(question, task_id, file_name).
    """
    def __init__(self):
        self.agent = CodeAgent(
            model = build_model(),
            tools = [
                DuckDuckGoSearchTool(),                         # Finds relevant pages (returns titles, URLs, snippets).
                VisitWebpageWithHint(max_output_length = 8000), # Like visit_webpage, plus a search_webpage hint when truncated.
                *FILE_TOOLS,                                    # Spreadsheets, code, audio
                *KNOWLEDGE_TOOLS,                               # Wikipedia, YouTube
            ],

            # Python modules the agent's generated code may import.
            additional_authorized_imports = ["re", "math", "json", "datetime", "statistics", "pandas", "openpyxl"],   
            max_steps = MAX_STEPS,
            executor_kwargs = {"timeout_seconds": 120},         # Allow 120 s (smolagents stops every code step after 30 s by default).
            planning_interval = PLANNING_INTERVAL,
            verbosity_level = 2,                                # Print the full Thought / Code / Observation trace
        )

    def __call__(self, question: str, task_id: str = "", file_name: str = "") -> str:
        task = question

        # If there is an attachment, tell the agent where it is and which tool fits.
        if file_name:
            path = ensure_file(task_id, file_name)
            if path is None:
                task += f"\n\n(An attached file '{file_name}' could not be downloaded.)"
            else:
                task += (
                    f"\n\nThe attached file is available locally at: {path.as_posix()}\n"
                    "Use the matching tool: read_spreadsheet (Excel/CSV), read_text_file or "
                    "run_python_file (.py), transcribe_audio (.mp3/.wav)."
                )
 
        task += f"\n\n{TOOL_HINTS}\n{ANSWER_RULES}"
 
        try:
            raw_answer = self.agent.run(task)
        except Exception as e:

            # Never crash the whole evaluation because of one question.
            print(f"[agent error] {e}")
            return ""
 
        answer = normalize_answer(raw_answer)

        # Garbage guard: code is never a valid GAIA answer (n empty answer is better).
        if CODE_PATTERN.search(answer):
            print(f"[agent warning] answer looks like code, discarded: {answer[:80]!r}")
            return ""

        return answer
    
