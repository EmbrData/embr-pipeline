# Egocentric data-quality eval (Qwen3-VL-8B judge)

Reproduce Build AI's `Egocentric-100K` quality benchmark (**hand visibility** +
**active manipulation**) on *your* clips, using an **open** VLM judge
(`Qwen3-VL-8B-Instruct`). No Gemini / Google key. Output is a deck-ready table +
bar chart.

## Install

```bash
pip install -r requirements.txt          # requests, imageio-ffmpeg, matplotlib, tqdm
```

`imageio-ffmpeg` ships a static ffmpeg, so you don't need a system ffmpeg.

## Pick a judge backend (verified 2026-06-30)

### A. HF Inference Providers — fastest path, zero infra (recommended for the 10K-frame eval)
`Qwen/Qwen3-VL-8B-Instruct` is served serverless via the HF router (provider: **Novita**)
on an **OpenAI-compatible** endpoint. Your **HF token is the API key** — create a
*fine-grained* token with the **"Make calls to Inference Providers"** permission.

```bash
export HF_TOKEN=hf_xxx
python run_eval.py --clips-dir ./clips \
    --base-url https://router.huggingface.co/v1 \
    --api-key  $HF_TOKEN \
    --model    Qwen/Qwen3-VL-8B-Instruct
```

Cost: a 10K-frame eval is **~$1.7** at Novita's $0.08/M-in + $0.50/M-out (≈1.2K in +
150 out tokens/frame). That fits inside **PRO's $2/mo** included credits; on the **Free**
tier ($0.10/mo) buy a few dollars of pay-as-you-go credits (HF bills at provider rates,
no markup). HF credits do **not** apply to HaMeR or Gemini.

### B. Local vLLM — best for batch / production (no per-call cost)
```bash
pip install vllm
# BF16 needs ~24 GB VRAM; the official FP8 checkpoint fits ~16 GB (vLLM/SGLang only):
vllm serve Qwen/Qwen3-VL-8B-Instruct            # or Qwen/Qwen3-VL-8B-Instruct-FP8
python run_eval.py --clips-dir ./clips \
    --base-url http://localhost:8000/v1 \
    --model    Qwen/Qwen3-VL-8B-Instruct
```

At ~1M frames, self-hosting FP8 on one GPU undercuts serverless (~$171/1M on Novita).

## Run

```bash
python run_eval.py --clips-dir ./clips --n-frames 10000 --out-dir eval_out
```

Outputs in `eval_out/`:
- `summary.json` — `hands_0/1plus/2_pct`, `active_pct`, counts
- `frames.csv` — per-frame judgments (resumable; cached in `cache.sqlite`)
- `table.md` — the slide table
- `benchmark.png` — grouped bar chart vs reference

Key flags: `--fps` (sampling rate within each clip, default 1), `--img-size` (long-side
px sent to the judge, default 768 — downscaling controls token cost), `--workers`,
`--seed`, `--no-glove-note`, `--your-label`, `--limit-clips`.

## ⚠️ Making the comparison valid (same-judge rule)

Absolute percentages depend on the judge. The built-in reference numbers are Build AI's
**published** figures, which they judged with **gemini-2.5-flash** — so a raw side-by-side
is *orientation only, not apples-to-apples*. To get a defensible comparison, re-judge the
reference frames with **this same script**, then pass the result:

```bash
# 1) get reference frames (Build AI's eval datasets are UNGATED):
#    - builddotai/Egocentric-100K-Evaluation  (egocentric_100k / ego4d / epic_kitchens parquets)
#    - Voxel51/Egocentric_10K_subset          (individual .mp4 files, easiest)
# 2) judge them with the SAME model:
python run_eval.py --clips-dir ./reference_clips --out-dir ref_out \
    --base-url https://router.huggingface.co/v1 --api-key $HF_TOKEN \
    --model Qwen/Qwen3-VL-8B-Instruct
# 3) judge your data and compare against the same-judge reference:
python run_eval.py --clips-dir ./clips --out-dir eval_out \
    --base-url https://router.huggingface.co/v1 --api-key $HF_TOKEN \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --reference-json ref_out/summary.json
```

Now both numbers come from the same judge on the same footing — stronger than parroting
Build AI's Gemini table.

## Gotchas (verified)

- **Gloves.** Factory workers wear gloves; an un-clarified judge under-counts gloved hands
  and you'll look *worse* than you are. The prompt adds "gloved hands count" by default;
  `--no-glove-note` matches Build AI's exact wording.
- **Downscale frames.** Image token cost is tile-based; large frames cost more. 768px
  long-side is a good default for hand counting.
- **Calibrate before committing budget.** Run a 100-frame pilot to measure real
  tokens/frame and throughput before a 1M-frame production run.
- **Prices are point-in-time (2026-06-30).** HF Free $0.10 is "subject to change" and the
  Qwen3-VL→Novita mapping is the most volatile bit — re-check at build time.

## Next stage: HaMeR (3D hand pose) — read before you build on it

HaMeR has **no hosted inference** (self-host only) and, more importantly, a
**non-commercial licensing chain**: HaMeR code is MIT, but its **MANO** dependency is
non-commercial (website registration for `MANO_RIGHT.pkl`), and the faster **WiLoR**
weights are **CC-BY-NC**. For a product you intend to **sell**, you must clear a
commercial MANO license or use a commercially-licensed hand-pose method. Safe path for the
demo: use HaMeR for **internal validation / quality scoring** only until licensing is
cleared. See the chat thread for alternatives.
