"""
Custom Tools for GAIA Attachments

Tools:
- read_spreadsheet  -> Excel / CSV files
- read_text_file    -> any text file (e.g., the attached .py script, to read its code)
- run_python_file   -> execute an attached Python script and return its output
- transcribe_audio  -> speech-to-text for .mp3 / .wav (local Whisper)
"""

import os
import subprocess
import sys 
from pathlib import Path

from smolagents import tool

MAX_CHARS = 10_000              # Maximum number of characters a tool returns (long outputs would flood the context window of the LLM).

def _truncate(text: str) -> str:
    """Helper (not a tool): cut long outputs and tell the agent."""
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS] + f"\n... [truncated, {len(text) - MAX_CHARS} more characters]"

    return text

@tool
def read_spreadsheet(path: str) -> str:
    """
    Reads an Excel (.xlsx, .xls) or CSV file and returns every sheet as CSV text.
    Use this for any attached spreadsheet before doing calculations on it.
    
    Args:
        path: Local path to the spreadsheet file.
    """
    import pandas as pd         # Import inside the function: the module is only loaded when the tool is used.

    try: 
        if path.lower().endswith(".csv"):
            sheets = {"csv": pd.read_csv(path)}
        else:
            sheets = pd.read_excel(path, sheet_name = None)
    except Exception as e:
        return f"Error reading spreadsheet '{path}': {e}"

    parts = []
    for name, df in sheets.items():
        # Shape and column names help the LLM understand the structure before it calculates.
        parts.append(f"### Sheet: {name} | {df.shape[0]} rows x {df.shape[1]} columns")
        parts.append(f"Columns: {list(df.columns)}")
        parts.append(df.to_csv(index = False))

    return _truncate("\n".join(parts))

@tool
def read_text_file(path: str) -> str:
    """
    Reads a plain-text file (e.g., .py, .txt, .json, .md) and returns its content.
    Use this to inspect the code or text of an attached file.

    Args:
        path: Local path to the text file.
    """
    try:
        # Never crash on unusual characters, replace them instead.
        return _truncate(Path(path).read_text(encoding = "utf-8", errors = "replace"))
    except Exception as e:
        return f"Error reading file '{path}: {e}"

@tool
def run_python_file(path: str) -> str:
    """
    Executes an attached Python script in a separate process and returns what it prints.
    Use this when a question asks for the output or result of a given Python file.

    Args:
        path: Local path to the .py file.
    """
    try:
        # Start a new Python process
        result = subprocess.run(
            [sys.executable, path],
            capture_output = True,              # Collect stdout/stderr instead of printing them
            text = True, 
            timeout = 60,                       # Kill scripts that run forever
        )
    except subprocess.TimeoutExpired:
        return "Error: the script did not finish within 60 seconds (possible endless loop)."
    except Exception as e:
        return f"Error running '{path}'; {e}"

    output = f"STDOUT:\n{result.stdout}"
    if result.stderr:
        output += f"\nSTDERR:\n{result.stderr}"
    output += f"\nExit code: {result.returncode}"

    return _truncate(output)

# Audio transcription: 
# Loading a Whisper model takes a few seconds and a lot of RAM, so it is loaded once and reused (the global starts as None and is filled on first use).
_whisper_model = None

@tool
def transcribe_audio(path: str) -> str:
    """
    Transcribes speech in an audio file (.mp3, .wav, .m4a) to text.
    Use this for any attached audio recording (e.g., voice memos, lectures).

    Args:
        path: Local path to the audio file.
    """
    global _whisper_model

    try:
        from faster_whisper import WhisperModel

        if _whisper_model is None:
            # Model size via .env: "small" is a good speed/quality balance on CPU, "medium" is more accurate but slower.
            # Downloads once, then cached.
            size = os.getenv("WHISPER_MODEL", "small")

            _whisper_model = WhisperModel(size, device = "cpu", compute_type="int8")    # Runs on any computer without a GPU

        segments, _info = _whisper_model.transcribe(path)       # Returns a generator of segments and info about the audio.
        text = " ".join(segment.text.strip() for segment in segments)

        return _truncate(text) if text else "The audio contained no recognizable speech."

    except Exception as e:
        return f"Error transcribing '{path}': {e}"

# Chess analysis: the LLM reads the board from the image and writes it as FEN, python-chess does the calculation.
last_chess_moves = None         # Moves reported by the last successfull chess_analyze call (read by the answer check in agent.py).

def _every_reply_allows_mate(board, moves_left: int) -> bool:
    """Helper (not a tool): after our move, True if every opponent reply still lets us force mate."""
    for reply in list(board.legal_moves):
        board.push(reply)
        ok = _can_force_mate(board, moves_left)
        board.pop()

        if not ok:
            return False                # One good defence is enough to refute our move.
    return True


def _can_force_mate(board, moves_left: int) -> bool:
    """Helper (not a tool): True if the side to move can force checkmate within `moves_left` moves."""
    for move in list(board.legal_moves):
        board.push(move)
        mated = board.is_checkmate() or (
            moves_left > 1 and not board.is_game_over()
            and _every_reply_allows_mate(board, moves_left - 1)
        )
        board.pop()                     # Always undo the move -> board is unchanged for the next try.

        if mated:
            return True
        
    return False

@tool
def chess_analyze(fen: str) -> str:
    """
    Analyzes a chess position and lists forced checkmates (in 1 or 2 moves) for the side to move.
    If there is none, it lists all checks and captures as candidate moves.
    Use this for chess questions: first write the position from the image as FEN, then call this.

    Args:
        fen: The position in FEN notation, e.g. "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1".
            Ranks from 8 down to 1, then "w" or "b" for the side to move.
    """
    global last_chess_moves         # Assign the module-level variable
    import chess

    try:
        board = chess.Board(fen)
    except ValueError as e:
        return f"Invalid FEN '{fen}': {e}"

    # Catch impossible positions (e.g. missing king, pawns on rank 1/8), which usually means the board was misread from the image.
    if not board.is_valid():
        return f"Illegal position ({board.status()!r}). Re-check the piece placement and the side to move."

    side = "White" if board.turn == chess.WHITE else "Black"
    mate_in_1, mate_in_2 = [], []

    for move in list(board.legal_moves):
        san = board.san(move)           # Algebraic notation (e.g. "Rd8#") must be computed before push.
        board.push(move)

        if board.is_checkmate():
            mate_in_1.append(san)
        elif not board.is_game_over() and _every_reply_allows_mate(board, 1):
            mate_in_2.append(san)
        board.pop()

    lines = [f"Position: {board.fen()}", f"Side to move: {side}"]
    if mate_in_1:
        lines.append(f"Checkmate in 1: {', '.join(mate_in_1)}")

    if mate_in_2:
        lines.append(f"Forced checkmate in 2, first move: {', '.join(mate_in_2)}")

    if not (mate_in_1 or mate_in_2):
        checks = [board.san(m) for m in board.legal_moves if board.gives_check(m)]
        captures = [board.san(m) for m in board.legal_moves if board.is_capture(m)]
        lines.append("No forced checkmate within 2 moves.")
        lines.append(f"Checks: {', '.join(checks) or 'none'}")
        lines.append(f"Captures: {', '.join(captures) or 'none'}")

    # Remember the result for the final-answer check (only reached with a valid position).
    last_chess_moves = mate_in_1 + mate_in_2

    return "\n".join(lines)

# Convenient list for agent.py: "from tools import FILE_TOOLS"
FILE_TOOLS = [read_spreadsheet, read_text_file, run_python_file, transcribe_audio, chess_analyze]

import re
import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"     # Wikimedia asks every client to send a descriptive User-Agent; anonymous ones may be blocked.
WIKI_HEADERS = {"User-Agent": "GAIA-course-agent/1.0 (educational project)"}
WIKI_MAX_CHARS = 20_000                             # Pages are long

@tool
def wikipedia_search(query: str) -> str:
    """
    Searches English Wikipedia and returns the top matching article titles with short snippets.
    Use this first to find the exact title of Wikipedia article, then read it with wikipedia_page.

    Args:
        query: What to search for (e.g., "Mercedes Sosa discography").
    """
    params = {
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": 5, "format": "json",
    }

    try:
        r = requests.get(WIKI_API, params = params, headers = WIKI_HEADERS, timeout = 20)
        r.raise_for_status()
        hits = r.json()["query"]["search"]
    except Exception as e:
        return f"Error searching Wikipedia: {e}"

    if not hits:
        return f"No Wikipedia articles found for '{query}'."

    lines = [f"- {h['title']}: {re.sub(r'<[^>]+>', '', h['snippet'])}" for h in hits]   # Snippets contain HTML highlight tags -> strip all tags.
    
    return "Top Wikipedia results:\n" + "\n".join(lines)

def _revision_as_of(title: str, as_of_date: str) -> int | None:
    """Helper (not a tool): id of the last revision of `title`on or before `as_of_date`."""
    params = {
        "action": "query", "prop": "revisions", "titles": title,
        "rvlimit": 1, "rvdir": "older",                 # Walk backwards in time.
        "rvstart": f"{as_of_date}T23:59:59Z",           # Start at the end of that day.
        "rvprop": "ids", "format": "json", "formatversion": 2,
    }
    r = requests.get(WIKI_API, params = params, headers = WIKI_HEADERS, timeout = 20)
    r.raise_for_status()
    page = r.json()["query"]["pages"][0]
    revisions = page.get("revisions")

    return revisions[0]["revid"] if revisions else None

def _wiki_html_to_markdown(html: str) -> str:
    """Helper (not a tool): turn Wikipedia HTML into compact Markdown for the LLM."""
    from bs4 import BeautifulSoup                       # HTML parser
    from markdownify import markdownify                 # Converts HTML to Markdown (keeps tables readable).

    soup = BeautifulSoup(html, "html.parser")

    # Remove page elements that carry no facts for the agent.
    for selector in [
        "span.mw-editsection",                          # "[edit]" links next to headings
        "sup.reference",                                # Citation markers like [1], [12]
        "figure", "img",                                # Images and their captions-in-figures
        "style", "script",                              # Embedded CSS/JS
        "div.reflist", "ol.references",                 # The reference list at the bottom
        "div.navbox", "table.navbox", "div.hatnote",    # Navigation boxes, "see also" notes
        "table.metadata", "div.mw-empty-elt",
    ]:
        for element in soup.select(selector):
            element.decompose()                         # Delete the element and everything inside it

    # Convert to Markdown (keep the link text but drop the URL).
    text = markdownify(str(soup), heading_style = "ATX", strip = ["a"])     # "## Heading" style

    # Cleanup of leftovers
    text = re.sub(r"\[\d+\]", "", text)                 # Any remaining [1]-style markers
    text = re.sub(r"[ \t]+\n", "\n", text)              # Trailing spaces at line ends
    text = re.sub(r"\n{3,}", "\n\n", text)              # Collapse many blank lines

    return text.strip()

@tool
def wikipedia_page(title: str, section: str = "", as_of_date: str = "") -> str:
    """
    Returns the content of an English Wikipedia article as Markdown, including tables.
    Use this to read facts, lists and tables (discographies, results, statistics) from Wikipedia.

    Args:
        title: Exact article title. e.g "Mercedes Sosa" (find it with wikipedia_search).
        section: Optional heading to jump to, e.g. "Discography". Leave empty for the whole page.
        as_of_date: Optional date YYYY-MM-DD to read the article as it was on that day, e.g. "2022-12-31" when a question refers to the 2022 version of Wikipedia.
    """
    try:
        params = {"action": "parse", "prop": "text", "format": "json", "formatversion": 2,"redirects": 1}
        
        if as_of_date:
            revid = _revision_as_of(title, as_of_date)

            if revid is None:
                return f"No revision of '{title}' exists on or before {as_of_date}."
            
            params["oldid"] = revid         # Parse that specific old revision
        else:
            params["page"] = title          # Parse the current version
 
        r = requests.get(WIKI_API, params = params, headers = WIKI_HEADERS, timeout = 30)
        r.raise_for_status()
        data = r.json()

        if "error" in data:

            # e.g. page does not exist -> readable message for the agent
            return f"Wikipedia error for '{title}': {data['error'].get('info')}"
        
        html = data["parse"]["text"]
    except Exception as e:
        return f"Error loading Wikipedia page '{title}': {e}"
 
    text = _wiki_html_to_markdown(html)

    if section:

        # Jump to the first Markdown heading that contains the requested section name.
        lines = text.splitlines()

        for i, line in enumerate(lines):

            if line.startswith("#") and section.lower() in line.lower():
                text = "\n".join(lines[i:])
                break
        else:
            headings = [l for l in lines if l.startswith("#")][:30]

            return f"Section '{section}' not found. Available headings:\n" + "\n".join(headings)

    if len(text) > WIKI_MAX_CHARS:
        text = text[:WIKI_MAX_CHARS] + (
            "\n... [truncated: use the 'section' argument to jump to a specific part]"
        )

    return text

@tool
def youtube_transcript(url: str) -> str:
    """
    Returns the spoken-text transcript (subtitles) of a Youtube video.
    Use this for questions about what is said in a video. It cannot describe what is only shown.

    Args:
        url: The Youtube video URL, e.g. "https://www.youtube.com/watch?v=abc123".
    """
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)     # Extract the 11-character video id from common URL formats (watch?v=, youtu.be/, shorts/).

    if not match:
        return f"Could not find a Youtube video id in '{url}'."
    video_id = match.group(1)

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()

        if hasattr(api, "fetch"):
            snippets = api.fetch(video_id)
            text = " ".join(s.text for s in snippets)
        else:
            text = " ".join(x["text"] for x in YouTubeTranscriptApi.get_transcript(video_id))
    except Exception as e:

        # Typical causes: video has no subtitles, or YouTube blocks the IP (often on cloud servers).
        return f"Could not get a transcript for video {video_id}: {e}"
 
    return _truncate(text)

# Browser-like but honest User-Agent: "compatible; <bot name>; +<info URL>" is the common bot convention.
WEB_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GAIA-course-agent/1.0; +https://huggingface.co/learn/agents-course)"}

def fetch_page_markdown(url: str) -> str:
    """Helper (not a tool): download a web page and return its readable text as Markdown (used by search_webpage and by visit_webpage in agent.py)."""
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    r = requests.get(url, headers = WEB_HEADERS, timeout = 30)
    r.raise_for_status()

    # Remove page parts without content (menus, scripts) -> shorter text, fewer false matches.
    soup = BeautifulSoup(r.text, "html.parser")
    for element in soup.select("script, style, nav, header, footer, noscript"):
        element.decompose()

    text = markdownify(str(soup), heading_style = "ATX", strip = ["a", "img"])
    return re.sub(r"\n{3,}", "\n\n", text).strip()          # Collapse many blank lines


@tool
def search_webpage(url: str, query: str) -> str:
    """Searches inside one web page and returns only the passages that contain the query terms.
    Use this for long pages (papers, lists, rosters, tables) where visit_webpage gets truncated before the part you need.

    Args:
        url: Full URL of the page, e.g. "https://en.wikipedia.org/wiki/Malko_Competition".
        query: Words to look for, e.g. "NASA award number" or "1977 walks". Case does not matter.
    """
    try:
        text = fetch_page_markdown(url)         # Shared helper: download + clean + Markdown
    except Exception as e:
        return f"Error loading '{url}': {e}"

    # Split into passages (paragraphs / table rows separated by blank lines).
    passages = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Score every passage: exact phrase = 10 points, each single keyword = 1 point.
    phrase = query.lower().strip()
    keywords = [w for w in re.findall(r"\w+", phrase) if len(w) > 2]
    scored = []

    for i, passage in enumerate(passages):
        low = passage.lower()
        score = (10 if phrase in low else 0) + sum(1 for w in keywords if w in low)

        if score:
            scored.append((score, i))

    if not scored:
        return f"No passage on {url} contains any of: {keywords}. Try other words or visit_webpage."

    # Best 5 passages, shown in page order, each with its neighbours for context.
    best = sorted(i for _, i in sorted(scored, reverse = True)[:5])
    blocks = []

    for i in best:
        context = "\n".join(passages[max(0, i - 1): i + 2])   # previous + match + next passage
        blocks.append(f"--- passage {i} of {len(passages)} ---\n{context}")

    return _truncate(f"Matches for '{query}' on {url}:\n\n" + "\n\n".join(blocks))

# Convenient list for agent.py: "from tools import KNOWLEDGE_TOOLS"
KNOWLEDGE_TOOLS = [wikipedia_search, wikipedia_page, youtube_transcript, search_webpage]

