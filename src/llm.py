"""Claude helpers: cost accounting with a hard budget, standard + batch calls."""
import json, os, re, time
import threading
import anthropic
from concurrent.futures import ThreadPoolExecutor

PRICE = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
BUDGET = float(os.environ.get("CLAUDE_BUDGET_USD", "120"))
LEDGER = os.path.join(os.environ.get("OUT_DIR", "/workspace/results"), "claude_spend.json")
client = anthropic.Anthropic(max_retries=8)
_lock = threading.Lock()


_total = [json.load(open(LEDGER))["usd"] if os.path.exists(LEDGER) else 0.0]


def spent():
    return _total[0]


def charge(model, usage, batch=False):
    pin, pout = PRICE[model]
    usd = (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6 * (0.5 if batch else 1.0)
    with _lock:
        _total[0] += usd
        tmp = LEDGER + ".tmp"
        json.dump({"usd": _total[0]}, open(tmp, "w"))
        os.replace(tmp, LEDGER)
    return usd


def check_budget(estimate=0.0):
    if spent() + estimate > BUDGET:
        raise SystemExit(f"Budget stop: spent ${spent():.2f} + est ${estimate:.2f} > ${BUDGET}")


def text_of(msg):
    return "".join(b.text for b in msg.content if b.type == "text")


def parse_json(s):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call(params):
    check_budget()
    with client.messages.stream(**params) as st:
        msg = st.get_final_message()
    charge(params["model"], msg.usage)
    return text_of(msg)


def call_many(params_list, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: _safe(call, p), params_list))


def _safe(f, p):
    try:
        return f(p)
    except SystemExit:
        raise
    except Exception as e:  # keep going; record failure
        return f"__ERROR__ {e!r}"


def call_batch(requests, poll=30, resume_id=None):
    """requests: list of (custom_id, params). Returns {custom_id: text}.
    resume_id: re-download results of an already-run batch (not charged again)."""
    if resume_id:
        b = client.messages.batches.retrieve(resume_id)
    else:
        check_budget()
        b = client.messages.batches.create(requests=[{"custom_id": c, "params": p} for c, p in requests])
    print("batch", b.id, flush=True)
    while True:
        b = client.messages.batches.retrieve(b.id)
        if b.processing_status == "ended":
            break
        print(" ", b.request_counts, flush=True)
        time.sleep(poll)
    out = {}
    for r in client.messages.batches.results(b.id):
        if r.result.type == "succeeded":
            m = r.result.message
            if not resume_id:
                charge(m.model if m.model in PRICE else requests[0][1]["model"], m.usage, batch=True)
            out[r.custom_id] = text_of(m)
        else:
            out[r.custom_id] = f"__ERROR__ {r.result.type}"
    print(f"spent so far ${spent():.2f}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(LEDGER), f"raw_{b.id}.json"), "w"))
    return out
