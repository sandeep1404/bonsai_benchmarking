import requests, time, json, random, string
from pathlib import Path

SERVER = "http://localhost:8080"
PROMPT_TOKENS = [256, 512, 1024, 2048]
GEN_TOKENS    = [128, 256, 512]

# NEW (replace with this):
NUM_REQUESTS  = 20
TAG = "maxn_bonsai1.7b"   # ← ONLY change this between runs
BASE_DIR      = Path(__file__).resolve().parent
OUTPUT_FILE   = BASE_DIR / "results" / f"profile_export_{TAG}.jsonl"

def make_prompt(n_tokens):
    # Generate a synthetic prompt of approximately n_tokens
    words = ["the", "a", "an", "is", "was", "are", "were", "be", "been",
             "have", "has", "do", "does", "will", "would", "could", "should"]
    return " ".join(random.choices(words, k=n_tokens))

def send_request(prompt, max_tokens):
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "ignore_eos": True   # Forces exact gen length
    }
    t_start = time.time_ns()
    r = requests.post(f"{SERVER}/v1/completions", json=payload, timeout=300)
    t_end = time.time_ns()
    data = r.json()
    return {
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "output_tokens": data["usage"]["completion_tokens"],
        "request_start_ns": t_start,
        "request_end_ns":   t_end,
        "latency_s": (t_end - t_start) / 1e9
    }

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results = []

with open(OUTPUT_FILE, "w") as f:
    for pt in PROMPT_TOKENS:
        for gt in GEN_TOKENS:
            print(f"\n=== prompt={pt} gen={gt} ===")
            combo_results = []
            for i in range(NUM_REQUESTS):
                prompt = make_prompt(pt)
                res = send_request(prompt, gt)
                res.update({"target_prompt_tokens": pt, "target_gen_tokens": gt, "run": i})
                f.write(json.dumps(res) + "\n")
                f.flush()
                combo_results.append(res)
                tok_s = res["output_tokens"] / res["latency_s"]
                print(f"  [{i+1:2d}/20] latency={res['latency_s']:.2f}s  tok/s={tok_s:.1f}")

            latencies = [r["latency_s"] for r in combo_results]
            avg_toks  = sum(r["output_tokens"] for r in combo_results) / len(combo_results)
            avg_tokps = sum(r["output_tokens"]/r["latency_s"] for r in combo_results) / len(combo_results)
            print(f"  → avg latency={sum(latencies)/len(latencies):.2f}s  avg tok/s={avg_tokps:.1f}")

print(f"\n✅ Done! Results saved to {OUTPUT_FILE}")
