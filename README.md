# GAIA Agent – Hugging Face Agents Course (Unit 4)

A smolagents CodeAgent that answers GAIA Level-1 questions (web research, Wikipedia, YouTube, audio, spreadsheets, Python files and images), 
built with free components only.

**Result:** 95 % (19/20) on the course evaluation set. The certificate threshold is 30 %.

## Architecture

```
questions -> run_local.py (answer cache) -> GaiaAgent -> CodeAgent (ReAct: Thought -> Code -> Observation)
                                                │            │
                                                │            ├─ Gemini free tier via LiteLLM
                                                │            │  (retries · fallback models · rate limiter)
                                                │            └─ 11 tools (below) + final_answer_checks
                                                ▼
                         normalize_answer -> garbage guard -> post-run check -> answers.json -> submit.py
```

| Tool | Purpose |
|---|---|
| `web_search` | DuckDuckGo search |
| `visit_webpage` | Page reader with a proper User-Agent and a "use search_webpage" hint when the page is truncated |
| `search_webpage` | Keyword retrieval inside long pages (top passages with context) |
| `wikipedia_search`, `wikipedia_page` | MediaWiki API, incl. sections and historical revisions (`as_of_date`) |
| `youtube_transcript` | Subtitles of YouTube videos |
| `transcribe_audio` | Local speech-to-text (faster-whisper, CPU) |
| `read_spreadsheet`, `read_text_file`, `run_python_file` | Excel/CSV, text files, sandboxed script execution (separate process) |
| `chess_analyze` | python-chess: forced mates / candidate moves from a FEN |

Images are passed directly to the multimodal model (`agent.run(task, images=[...])`).

## Design decisions

- Exact match: answer rules appended to every task and a `normalize_answer()` post-processor.
- LLM translates, code computes: pandas for spreadsheets, python-chess for chess positions.
- Prompts are requests, code is rules: `final_answer_checks` reject answers that are not
  grounded in tool output, and every answer is checked again after the run (answers forced by "max steps" bypass smolagents' checks).
- Just-in-time hints: instructions appear inside tool observations where they are needed.
- Empty beats wrong: a garbage guard discards code-like answers.
- Free tier friendly: answer cache, rate limiter, retries, per-model fallback quotas.

## Lessons learned

- Most failures came from plumbing, not from the model: an unregistered tool, a missing `images=` argument, a 403 caused by the default User-Agent.
- Weak models ignore prompt rules; must-have behaviour has to be enforced in code.
- LLMs can fabricate tool observations (invent outputs of code they never ran), so answers should be traceable to real tool results in the execution log.
- One benchmark run is a sample: outputs vary between runs.

## Run locally (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# .env (never commit it):
#   MODEL_ID=gemini/<model>          GEMINI_API_KEY=...
#   FALLBACK_MODELS=gemini/<a>,gemini/<b>
#   REQUESTS_PER_MINUTE=15           NUM_RETRIES=4
#   HF_USERNAME=<hf-name>            CODE_URL=<this repo>

python check_setup.py          # environment + one LLM test call
python fetch_questions.py      # questions + attachments into data/
python test_tools.py           # tools without the LLM
python run_local.py            # all questions (cached in data/answers.json)
python run_local.py --show     # answer table
python submit.py               # dry run
python submit.py --send        # submit to the course leaderboard
```

Attachments that the course API cannot serve are loaded from the gated `gaia-benchmark/GAIA`
dataset (attachment files only, never the metadata).

## Stack

smolagents · LiteLLM · Google Gemini (free tier) · ddgs · BeautifulSoup · markdownify ·
pandas · openpyxl · faster-whisper · youtube-transcript-api · python-chess · Pillow