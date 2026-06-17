"""
quiz_generator.py
─────────────────────────────────────────────────────────────────────────────
Responsibilities:
  1. Initialise the Groq SDK client (singleton, loaded once).
  2. Accept a selected PromptStyle + user topic, retrieve relevant context
     from ChromaDB via rag_engine, and call the Groq API.
  3. Stream the response token-by-token so Streamlit can render it live.
  4. Enforce a token budget guard to avoid hitting free-tier rate limits.
  5. Expose a clean public function:  generate_quiz(display_name, topic)
     that yields text chunks for a Streamlit st.write_stream() call.

Free-tier Groq limits (llama-3.1-8b-instant as of 2024):
  • 30 requests / minute
  • 14,400 requests / day
  • 131,072 context tokens / request   ← generous; we stay well under
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import time
from typing import Generator, Iterator

from dotenv import load_dotenv
from groq import Groq, APIError, APIConnectionError, RateLimitError

from prompt_styles import PromptStyle, build_messages, get_style
from rag_engine import retrieve_context

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()  # reads GROQ_API_KEY from .env

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
GROQ_MODEL         = "llama-3.1-8b-instant"
MAX_OUTPUT_TOKENS  = 2048    # enough for all 4 styles; keeps latency low
TEMPERATURE        = 0.7     # balanced creativity ↔ accuracy
TOP_P              = 0.9
RETRIEVAL_TOP_K    = 6       # chunks pulled from ChromaDB per query

# ── Rate-limit guard ───────────────────────────────────────────────────────────
# Track the last request time so we can warn the user if they fire too fast.
_last_request_ts: float = 0.0
MIN_REQUEST_GAP_SEC = 3      # soft floor; Groq allows 30 RPM on free tier


# ══════════════════════════════════════════════════════════════════════════════
# Client singleton
# ══════════════════════════════════════════════════════════════════════════════

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    """
    Lazily initialise the Groq SDK client.
    Raises a clear RuntimeError if the API key is missing rather than
    letting the SDK throw a cryptic auth error.
    """
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file:  GROQ_API_KEY=gsk_..."
            )
        _groq_client = Groq(api_key=api_key)
        log.info("Groq client initialised (model: %s).", GROQ_MODEL)
    return _groq_client


# ══════════════════════════════════════════════════════════════════════════════
# Token budget estimator
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 characters per token (good enough for a guard).
    Groq's tokeniser is BPE-based; this is deliberately conservative.
    """
    return len(text) // 4


def _check_token_budget(messages: list[dict]) -> None:
    """
    Raise ValueError if the assembled prompt likely exceeds a safe threshold.
    We target ≤ 4 000 prompt tokens so MAX_OUTPUT_TOKENS stays comfortable
    within the 131 072 context window and free-tier usage stays low.
    """
    prompt_text = " ".join(m["content"] for m in messages)
    estimated   = _estimate_tokens(prompt_text)
    budget      = 4_000  # prompt token soft cap

    if estimated > budget:
        raise ValueError(
            f"Prompt is too large (~{estimated} estimated tokens). "
            f"Try uploading a shorter PDF or reducing the number of retrieved "
            f"chunks (current: {RETRIEVAL_TOP_K}). "
            f"Soft cap: {budget} prompt tokens."
        )
    log.info("Estimated prompt tokens: ~%d (budget: %d).", estimated, budget)


# ══════════════════════════════════════════════════════════════════════════════
# Rate-limit soft guard
# ══════════════════════════════════════════════════════════════════════════════

def _enforce_request_gap() -> None:
    """
    If requests come in faster than MIN_REQUEST_GAP_SEC apart, sleep briefly.
    This is a courtesy to the free tier, not a hard limiter.
    """
    global _last_request_ts
    elapsed = time.monotonic() - _last_request_ts
    if elapsed < MIN_REQUEST_GAP_SEC and _last_request_ts > 0:
        sleep_for = MIN_REQUEST_GAP_SEC - elapsed
        log.info("Rate-limit guard: sleeping %.1f s before next request.", sleep_for)
        time.sleep(sleep_for)
    _last_request_ts = time.monotonic()


# ══════════════════════════════════════════════════════════════════════════════
# Core streaming call
# ══════════════════════════════════════════════════════════════════════════════

def _stream_completion(
    messages: list[dict],
) -> Generator[str, None, None]:
    """
    Call the Groq API with streaming enabled and yield text deltas.

    Groq's streaming API returns server-sent events; the SDK surfaces them
    as an iterable of ChatCompletionChunk objects. We extract only the
    text delta from each chunk and yield it immediately so Streamlit can
    render it progressively without waiting for the full response.

    Yields:
        str — incremental text tokens as they arrive.

    Raises:
        RateLimitError  — hit the free-tier RPM/day limit.
        APIConnectionError — network issue reaching Groq.
        APIError        — any other Groq API-level error.
    """
    client = _get_groq_client()
    _enforce_request_gap()

    log.info("Sending request to Groq (%s) …", GROQ_MODEL)

    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stream=True,              # ← key: enables token-by-token delivery
        )

        for chunk in stream:
            # Each chunk has a list of choices; we only ever use choice[0]
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except RateLimitError as exc:
        log.error("Groq rate limit hit: %s", exc)
        raise RuntimeError(
            "🚦 Groq free-tier rate limit reached. "
            "Wait ~60 seconds and try again, or check your usage at "
            "console.groq.com."
        ) from exc

    except APIConnectionError as exc:
        log.error("Groq connection error: %s", exc)
        raise RuntimeError(
            "🔌 Could not reach the Groq API. "
            "Check your internet connection and try again."
        ) from exc

    except APIError as exc:
        log.error("Groq API error (status %s): %s", exc.status_code, exc)
        raise RuntimeError(
            f"⚠️ Groq API returned an error (HTTP {exc.status_code}). "
            f"Details: {exc.message}"
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_quiz(
    display_name: str,
    topic: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> Iterator[str]:
    """
    Full RAG-to-quiz pipeline, exposed as a streaming generator.

    This is the only function app.py needs to import from this module.

    Pipeline:
      1. Resolve the display_name → PromptStyle dataclass.
      2. Embed the topic query → retrieve top-k chunks from ChromaDB.
      3. Inject context into the style's prompt template.
      4. Check token budget.
      5. Stream Groq response → yield text deltas.

    Usage in Streamlit:
        with st.spinner("Generating…"):
            st.write_stream(generate_quiz(selected_style, topic_text))

    Args:
        display_name: Exact string from prompt_styles.STYLE_DISPLAY_NAMES
                      (e.g. "🎓 VTU Exam Style").
        topic:        The subject/topic the user typed in the UI.
        top_k:        How many RAG chunks to retrieve (default 6).

    Yields:
        str — incremental quiz text tokens.

    Raises:
        RuntimeError: on API key issues, rate limits, or network errors.
        ValueError:   if the assembled prompt exceeds the token budget.
        KeyError:     if display_name is not a registered style.
    """
    if not topic.strip():
        raise ValueError(
            "Topic cannot be empty. "
            "Please describe what concepts you want to be quizzed on."
        )

    # ── Step 1: resolve style ──────────────────────────────────────────────
    style: PromptStyle = get_style(display_name)
    log.info("Style selected: '%s'", style.key)

    # ── Step 2: retrieve context from ChromaDB ─────────────────────────────
    log.info("Retrieving context for topic: '%s'", topic[:80])
    context = retrieve_context(query=topic, top_k=top_k)

    if not context.strip():
        raise RuntimeError(
            "No relevant content found in the uploaded PDF for that topic. "
            "Try rephrasing the topic or uploading a different document."
        )

    # ── Step 3: build message list ─────────────────────────────────────────
    messages = build_messages(style=style, context=context, topic=topic)

    # ── Step 4: token budget guard ─────────────────────────────────────────
    _check_token_budget(messages)

    # ── Step 5: stream from Groq ───────────────────────────────────────────
    yield from _stream_completion(messages)


# ══════════════════════════════════════════════════════════════════════════════
# CLI smoke-test  (python quiz_generator.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from prompt_styles import STYLE_DISPLAY_NAMES

    print("Available styles:")
    for i, name in enumerate(STYLE_DISPLAY_NAMES, 1):
        print(f"  {i}. {name}")

    # Requires rag_engine to already have an ingested collection.
    # Run:  python rag_engine.py notes.pdf   first.
    test_style = STYLE_DISPLAY_NAMES[0]   # VTU by default
    test_topic = sys.argv[1] if len(sys.argv) > 1 else "Operating System Scheduling"

    print(f"\nGenerating quiz | Style: '{test_style}' | Topic: '{test_topic}'\n")
    print("─" * 70)

    try:
        for chunk in generate_quiz(test_style, test_topic):
            print(chunk, end="", flush=True)
        print("\n" + "─" * 70)
    except Exception as e:
        print(f"\n[ERROR] {e}")