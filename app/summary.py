"""LLM summary layer (PLAN G): runs after all four results exist, prose only, local model via Ollama.

The model may only restate the structured results. Output is JSON-validated; anything that diagnoses, stages, adds a number that
is not in the input, or combines modalities numerically is rejected and a fixed template is used instead."""
import json
import re
import urllib.request

from .combine import combine
from .schema import ORDER

BANNED = re.compile(r"\b(diagnos\w*|stage\s*\d|early[- ]stage|late[- ]stage|mild|moderate|severe|severity|prognos\w*|"
                    r"likelihood|risk score|overall (score|accuracy)|probability of)\b", re.I)


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["modalities", "overall"],
          "properties": {"modalities": {"type": "object", "additionalProperties": False, "required": list(ORDER),
                                        "properties": {m: {"type": "string"} for m in ORDER}},
                         "overall": {"type": "string"}}}   # passed to Ollama as `format`: the shape is enforced, the wording still validated


def template(results, combo):
    by = {r.modality: r for r in results}
    parts = {}
    for m in ORDER:
        r = by.get(m)
        parts[m] = "Not run." if r is None or r.status == "not_run" else f"{r.label}. {r.evidence} {r.caveat}"
    return {"modalities": parts, "overall": combo["text"], "source": "template"}


def build_prompt(results, combo):
    data = {"results": [r.to_dict() for r in results], "rule_outcome": combo}
    return ("You write a short plain-language note for a clinician from the structured screening results below. "
            "Rules: restate only what is in the data; one short paragraph per modality in the order clock, motor, eye, speech; "
            "do not diagnose, name stages or severity, add numbers, or combine modalities into one score; mention each caveat. "
            "Reply with JSON only: {\"modalities\": {\"clock\": str, \"motor\": str, \"eye\": str, \"speech\": str}, \"overall\": str}.\n\n"
            + json.dumps(data, indent=1))


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def validate(reply, results, combo):
    """Return the parsed reply if it obeys the rules, else raise ValueError."""
    d = json.loads(reply)
    if set(d) != {"modalities", "overall"} or set(d["modalities"]) != set(ORDER):
        raise ValueError("wrong JSON shape")
    texts = list(d["modalities"].values()) + [d["overall"]]
    if not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError("empty or non-string field")
    joined = " ".join(texts)
    if BANNED.search(re.sub(r"(not an?|no) diagnos\w*", "", joined, flags=re.I)):   # the rule text itself says "not a diagnosis"
        raise ValueError("diagnostic, staging or fusion wording")
    allowed = _numbers(json.dumps([r.to_dict() for r in results]) + json.dumps(combo))
    extra = _numbers(joined) - allowed
    if extra:
        raise ValueError(f"numbers not in the input: {sorted(extra)}")
    return d


def summarize(results, model="llama3.1:8b", url="http://localhost:11434/api/generate", timeout=120, post=None):
    combo = combine(results)
    if any(r.status == "not_run" for r in results) or {r.modality for r in results} != set(ORDER):
        out = template(results, combo)
        out["note"] = "not all four screens completed; template used"
        return out
    post = post or _ollama
    try:
        reply = post(url, {"model": model, "prompt": build_prompt(results, combo), "stream": False, "format": SCHEMA}, timeout)
        d = validate(reply, results, combo)
        d["overall"] = combo["text"]   # the rule outcome is fixed text, never model-written (it carries the "no accuracy claim" line)
        d["source"] = model
        return d
    except Exception as e:  # unreachable model, invalid JSON, or a rule violation: fall back, never show the model's text
        out = template(results, combo)
        out["note"] = f"LLM output not used ({type(e).__name__}: {e})"
        return out


def _ollama(url, payload, timeout):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["response"]
