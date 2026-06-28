# Bonsai LLM Benchmark — Formulas & Concepts
> **Device:** NVIDIA Jetson Orin Nano Super 8GB  
> **Stack:** JetPack 6.2.1 · L4T R36.4.3 · CUDA 12.6  
> **Purpose:** Personal reference for understanding benchmark metrics from the [SmolHub Bonsai Benchmark article](https://www.smolhub.com/posts/jetson-orin-nano-super-bonsai-benchmark/)

---

## 🍪 The Big Picture (Simple Analogy)

Think of your Orin Nano as a **cookie-making machine**:
- You feed it **ingredients** → these are the **prompt tokens** (your input/question)
- It **bakes and outputs cookies** → these are the **generated tokens** (the model's reply)

The benchmark measures:
- ⚡ **How fast** it makes cookies (throughput)
- 🕐 **How long before the first cookie appears** (TTFT)
- 🔋 **How many cookies per unit of electricity** (energy efficiency)

---

## 1. Throughput — `tok/s`

### What it measures
How many tokens (words/pieces) the model **generates per second**. The headline speed metric.

### Formula

$$
\text{Throughput (tok/s)} = \frac{\text{Total tokens generated}}{\text{Total time (seconds)}}
$$

### Example
> Model generates **256 tokens** in **8 seconds**
>
> Throughput = 256 ÷ 8 = **32 tok/s**

### Rules of thumb
- Higher = better
- MAXN_SUPER mode: ~38 tok/s (Ternary-Bonsai-1.7B)
- 25W mode: ~34.7 tok/s ← sweet spot
- 15W mode: ~23.4 tok/s

---

## 2. Time to First Token (TTFT) — `ms`

### What it measures
The delay from when you **send your prompt** to when the **very first output token** appears. This is the **prefill phase** — the GPU is digesting your entire input before producing anything.

### Formula

$$
\text{TTFT} = T_{\text{first output token}} - T_{\text{request sent}}
$$

### Example
> You send a 1024-token prompt at **t = 0 ms**  
> First output token appears at **t = 420 ms**  
> TTFT = **420 ms**

### Key insight
- Longer prompts → higher TTFT (more input to process)
- Short prompts (256 tok) → very low TTFT
- Lower = better

---

## 3. Inter-Token Latency (ITL) — `ms/tok`

### What it measures
Once the first token is out, how many **milliseconds between each subsequent token**? This is the **decode phase** — the steady rhythm of generation.

### Formula

$$
\text{ITL} = \frac{\text{Total generation time} - \text{TTFT}}{\text{Tokens generated} - 1}
$$

### Example
> TTFT = 420 ms  
> Total generation time = 4220 ms  
> Tokens generated = 128  
>
> ITL = (4220 - 420) ÷ (128 - 1) = 3800 ÷ 127 ≈ **~30 ms/tok**

### Relationship to throughput
ITL and throughput are inverses:

$$
\text{Throughput (tok/s)} \approx \frac{1000}{\text{ITL (ms)}}
$$

> If ITL = 30 ms → Throughput ≈ 1000 ÷ 30 ≈ **33 tok/s**

---

## 4. Energy per Token — `J/tok`

### What it measures
How many **joules of electrical energy** are consumed to produce a single token. Lower = more efficient.

### Formula

$$
\text{J/tok} = \frac{\text{Power (W)} \times \text{Time (s)}}{\text{Tokens generated}}
$$

### Example
> Power mode: **25W**  
> Generates **256 tokens** in **8 seconds**  
>
> Energy = 25 × 8 = 200 Joules  
> J/tok = 200 ÷ 256 = **~0.78 J/tok**

---

## 5. Tokens per Joule — `tok/J`

### What it measures
The **flip side of J/tok** — how many tokens do you get per joule? Higher = more efficient. This is the "bang for the buck" metric.

### Formula

$$
\text{tok/J} = \frac{1}{\text{J/tok}} = \frac{\text{Tokens generated}}{\text{Power (W)} \times \text{Time (s)}}
$$

### Example from the benchmark

| Power Mode | Throughput (tok/s) | tok/J | Verdict |
|------------|-------------------|-------|---------|
| 15W        | 23.4              | 4.94  | Efficient |
| **25W**    | **34.7**          | **5.18** | ✅ **Sweet spot** |
| MAXN_SUPER | 38.0              | 4.55  | Fastest but wasteful |

> **Key insight:** 25W gives you 47% more speed than 15W while actually being *more* energy-efficient than MAXN_SUPER.

---

## 6. P95 Latency — `ms`

### What it measures
If you run **20 requests**, sort all their response times from fastest to slowest, P95 is the time at the **95th percentile** — meaning 95% of your requests finish **at or below** this time. It gives a realistic "worst-case normal" without being distorted by rare spikes.

### How it works (visual)

```
20 requests sorted by latency (ms):
[310, 315, 318, 320, 322, 325, 328, 330, 333, 335,
 338, 340, 345, 350, 355, 360, 370, 385, 400, 850]
                                              ^^^
                                        outlier (thermal hiccup?)

P95 = value at position 19 out of 20 = ~400 ms
The 850 ms spike is ignored for practical purposes.
```

### Formula

$$
\text{P95 position} = \lceil 0.95 \times N \rceil \text{ where } N = \text{number of requests}
$$

> For 20 requests: position = ⌈0.95 × 20⌉ = **position 19**

---

## 7. The Benchmark Sweep Explained

### What is a sweep?
Running **every combination** of variables automatically — like a multiplication table.

### The formula from the article

```
prompt ∈ {256, 512, 1024, 2048} tokens   →  4 prompt lengths
×
gen    ∈ {128, 256, 512}        tokens   →  3 generation lengths
×
                                20 requests per combination
```

**Total requests per model per power mode = 4 × 3 × 20 = 240 requests**

### What each variable tests

| Variable | Tests | Bottleneck |
|----------|-------|------------|
| Short prompt + Long gen | Output speed | Memory bandwidth (decode) |
| Long prompt + Short gen | Input processing | Compute (prefill) |
| 20 repeats per combo | Statistical reliability | Filters noise/variance |

### The llama-bench command

```bash
./build/bin/llama-bench \
  -m ~/models/ternary-1.7b-q2/*.gguf \
  -p 256,512,1024,2048 \   # prompt sweep
  -n 128,256,512 \         # generation sweep
  -ngl 99 \                # all layers on GPU
  -r 20                    # 20 requests per combo
```

---

## 8. Full Worked Example (End-to-End)

**Setup:** Ternary-Bonsai-1.7B · 25W mode · prompt=512 tok · gen=256 tok

| Metric | Calculation | Result |
|--------|------------|--------|
| TTFT | Time to first token | ~180 ms |
| Generation time | After first token | ~7,380 ms |
| ITL | 7380 ÷ (256-1) | ~29 ms/tok |
| Throughput | 1000 ÷ 29 | ~34.5 tok/s |
| Energy used | 25W × 7.56s | ~189 J |
| J/tok | 189 ÷ 256 | ~0.74 J/tok |
| **tok/J** | **1 ÷ 0.74** | **~5.18 tok/J ✅** |
| P95 latency | 95th percentile of 20 runs | ~420 ms |

---

## 9. Quick Reference Card

| Metric | Formula | Unit | Better when |
|--------|---------|------|-------------|
| Throughput | tokens ÷ time | tok/s | ⬆ Higher |
| TTFT | first_token_time − send_time | ms | ⬇ Lower |
| ITL | (gen_time − TTFT) ÷ (n_tokens − 1) | ms/tok | ⬇ Lower |
| J/tok | (power × time) ÷ tokens | J/tok | ⬇ Lower |
| tok/J | tokens ÷ (power × time) | tok/J | ⬆ Higher |
| P95 | 95th percentile latency over N runs | ms | ⬇ Lower |

---

## 10. Power Modes on Your Board

```bash
# Check current mode
sudo nvpmodel -q --verbose

# Switch modes
sudo nvpmodel -m 0    # 15W
sudo nvpmodel -m 1    # 25W  ← recommended sweet spot
sudo nvpmodel -m 2    # MAXN_SUPER

# Lock clocks after switching (important for fair benchmarking!)
sudo jetson_clocks
```

> ⚠️ Note: Your Orin Nano Super has **3 modes** (15W, 25W, MAXN_SUPER). The 7W mode only exists on the older non-Super flash configuration and is not available on your board — this is expected and correct.

---

*Notes compiled from: [SmolHub Bonsai Benchmark](https://www.smolhub.com/posts/jetson-orin-nano-super-bonsai-benchmark/) · NVIDIA JetPack 6.2.1 · L4T R36.4.3*
