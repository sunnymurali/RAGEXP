"""Azure OpenAI access: chat (Responses API) and embeddings, with per-call tracing (latency, tokens, prompt/response)."""
from __future__ import annotations
import functools, json, logging, os, re, time
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("rag.llm")
TRACE_PAYLOADS = os.getenv("TRACE_PAYLOADS", "1") != "0"   # keep prompts/responses in run records (always in logs/rag.log at DEBUG)

def chat_deployment() -> str:
    return os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")

def embedding_deployment() -> str:
    return os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")

def _require(settings: dict[str, str], what: str) -> dict[str, str]:
    missing = [name for name, value in settings.items() if not value]
    if missing: raise RuntimeError(f"{what} is not configured; set {', '.join(missing)} in .env")
    return settings

@functools.lru_cache(maxsize=1)
def get_chat():
    from langchain_openai import AzureChatOpenAI
    c = _require({
        "AZURE_OPENAI_CHAT_DEPLOYMENT": chat_deployment(),
        "AZURE_OPENAI_CHAT_ENDPOINT": os.getenv("AZURE_OPENAI_CHAT_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "AZURE_OPENAI_CHAT_API_KEY": os.getenv("AZURE_OPENAI_CHAT_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY", ""),
        "AZURE_OPENAI_CHAT_API_VERSION": os.getenv("AZURE_OPENAI_CHAT_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", ""),
    }, "Chat model")
    log.info("Chat model: %s @ %s (Responses API)", c["AZURE_OPENAI_CHAT_DEPLOYMENT"], c["AZURE_OPENAI_CHAT_ENDPOINT"])
    return AzureChatOpenAI(azure_deployment=c["AZURE_OPENAI_CHAT_DEPLOYMENT"], azure_endpoint=c["AZURE_OPENAI_CHAT_ENDPOINT"],
                           api_key=c["AZURE_OPENAI_CHAT_API_KEY"], api_version=c["AZURE_OPENAI_CHAT_API_VERSION"],
                           use_responses_api=True, timeout=90, max_retries=2)

@functools.lru_cache(maxsize=1)
def get_embeddings():
    from langchain_openai import AzureOpenAIEmbeddings
    c = _require({
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": embedding_deployment(),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", ""),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", ""),
    }, "Embedding model")
    log.info("Embedding model: %s @ %s", c["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"], c["AZURE_OPENAI_ENDPOINT"])
    # max_retries=0: store.py owns rate-limit backoff so it can report progress.
    return AzureOpenAIEmbeddings(azure_deployment=c["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"], azure_endpoint=c["AZURE_OPENAI_ENDPOINT"],
                                 api_key=c["AZURE_OPENAI_API_KEY"], api_version=c["AZURE_OPENAI_API_VERSION"], max_retries=0)

def chat_text(msg) -> str:
    """Responses API returns content blocks; chat completions return a string."""
    c = msg.content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c) if isinstance(c, list) else str(c)

def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}

def call_chat(purpose: str, system: str, user: str) -> tuple[str, dict]:
    """One traced chat call. Returns (text, info) where info = latency, token usage and (optionally) the prompt/response."""
    t0 = time.perf_counter()
    log.debug("LLM[%s] request | system:\n%s\n--- user:\n%s", purpose, system, user)
    try:
        msg = get_chat().invoke([("system", system), ("human", user)])
    except Exception as e:
        log.error("LLM[%s] failed after %.0f ms: %s: %s", purpose, (time.perf_counter() - t0) * 1000, type(e).__name__, str(e)[:300])
        raise
    ms = round((time.perf_counter() - t0) * 1000); text = chat_text(msg); usage = getattr(msg, "usage_metadata", None) or {}
    info = {"purpose": purpose, "model": chat_deployment(), "ms": ms, "prompt_chars": len(system) + len(user), "response_chars": len(text),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": (usage.get("output_token_details") or {}).get("reasoning")}
    if TRACE_PAYLOADS: info.update(system=system, user=user, response=text)
    log.info("LLM[%s] %d ms | tokens in=%s out=%s reasoning=%s | prompt %d chars -> response %d chars", purpose, ms,
             info["input_tokens"], info["output_tokens"], info["reasoning_tokens"], info["prompt_chars"], info["response_chars"])
    log.debug("LLM[%s] response:\n%s", purpose, text)
    return text, info
