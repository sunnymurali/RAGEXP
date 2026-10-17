"""Answer verification: a weighted 0-1 score from grounding checks plus an LLM faithfulness judge."""
from __future__ import annotations
import logging, re
import llm

log = logging.getLogger("rag.verify")

WEIGHTS = {"judge": .40, "numbers": .20, "metric": .15, "period": .10, "citation": .10, "rank": .05}
LOW_SCORE, HIGH_SCORE = 0.5, 0.8

JUDGE_PROMPT = ("You are a strict fact-checker. Break the ANSWER into atomic factual claims and decide for each whether the EVIDENCE "
                "explicitly supports it (numbers, units, dates and entities must match). Ignore citation markers like [p6]. "
                'Return ONLY JSON: {"total_claims": n, "supported_claims": n, "unsupported": ["claim", ...]}.')

Component = tuple[float | None, str]   # (score or None when not applicable, human-readable detail)

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()

def contains_any(text: str, terms: list[str]) -> bool:
    n = norm(text); return any(t.lower() in n for t in terms if t)

def _judge(question: str, answer: str, evidence: str) -> tuple[Component, dict | None]:
    try:
        raw, info = llm.call_chat("judge", JUDGE_PROMPT, f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}\nANSWER: {answer}")
        d = llm.parse_json(raw); total, supported = int(d.get("total_claims", 0)), int(d.get("supported_claims", 0))
        if total <= 0: return (None, "judge found no claims"), info
        unsupported = d.get("unsupported") or []
        return (max(0.0, min(1.0, supported / total)), f"{supported}/{total} claims supported" + (f"; unsupported: {unsupported}" if unsupported else "")), info
    except Exception as e:
        return (None, f"judge failed: {type(e).__name__}"), None

def verify_answer(s: dict, answer: str, used: list[dict]) -> tuple[dict, list[dict]]:
    """Score `answer` against the cited evidence passages `used`. Components that don't apply are dropped and weights renormalised.
    Returns (verification, llm_call_infos)."""
    ev_text = " ".join(x["passage"]["text"] for x in used)
    comps: dict[str, Component] = {}

    numbers = sorted({n.rstrip(",.") for n in re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?", re.sub(r"\[[^\]]*\]", "", answer))})
    missing = [n for n in numbers if n not in ev_text]
    comps["numbers"] = (((len(numbers) - len(missing)) / len(numbers), f"{len(numbers) - len(missing)}/{len(numbers)} numbers found verbatim" + (f"; missing {missing}" if missing else ""))
                        if numbers else (None, "no numbers in answer"))

    comps["metric"] = ((1.0, f"metric {s['metrics']} present in cited evidence") if contains_any(ev_text, s["metrics"])
                       else (0.0, f"metric {s['metrics']} NOT in cited evidence")) if s["metrics"] else (None, "no metric extracted")

    if s["period_terms"]:
        found = contains_any(ev_text, s["period_terms"])
        comps["period"] = (1.0 if found else 0.0, f"period {s['period_terms']} " + ("found" if found else "NOT found"))
    else:
        comps["period"] = (None, "no period asked")

    evidence_ids = [x["passage"]["chunk_id"] for x in s["evidence"]]
    cited = set(re.findall(r"\bp\d+(?:\.\d+)?\b", answer))
    comps["citation"] = ((len(cited & set(evidence_ids)) / len(cited), f"answer cites {sorted(cited)}") if cited else (0.0, "answer has no explicit chunk citation"))

    ranks = [evidence_ids.index(x["passage"]["chunk_id"]) for x in used if x["passage"]["chunk_id"] in evidence_ids]
    comps["rank"] = (1 / (1 + min(ranks)), f"best cited chunk is rank {min(ranks) + 1} of {len(evidence_ids)}") if ranks else (None, "n/a")

    calls: list[dict] = []
    if s.get("judge", True) and used:
        comps["judge"], info = _judge(s["question"], answer, "\n\n".join(f"[{x['passage']['chunk_id']}] {x['passage']['text']}" for x in used))
        if info: calls.append(info)
    else:
        comps["judge"] = (None, "disabled")

    live = {k: v for k, v in comps.items() if v[0] is not None}
    total_weight = sum(WEIGHTS[k] for k in live)
    score = sum(WEIGHTS[k] * v[0] for k, v in live.items()) / total_weight if total_weight else 0.0
    label = "high" if score >= HIGH_SCORE else ("medium" if score >= LOW_SCORE else "low")
    verification = {"score": round(score, 3), "label": label,
                    "components": {k: {"score": None if v[0] is None else round(v[0], 3), "weight": WEIGHTS[k], "detail": v[1]} for k, v in comps.items()}}
    return verification, calls
