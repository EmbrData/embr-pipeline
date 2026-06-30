#!/usr/bin/env python3
"""
run_eval.py - Egocentric data-quality benchmark with an OPEN VLM judge (Qwen3-VL-8B).

Reproduces Build AI's Egocentric-100K eval (hand-visibility + active-manipulation)
on YOUR clips, using Qwen3-VL-8B-Instruct as the judge. No Gemini / Google key.

What it does
  1. Sample frames from your clips (ffmpeg, ~--fps fps, allocated across clips
     proportional to duration so long clips don't dominate -> unbiased ~N frames).
  2. Judge each frame with two structured-output prompts (verbatim from Build AI):
       hand_count          -> {0, 1, 2}
       active_manipulation -> {yes, no}
  3. Aggregate -> summary.json + frames.csv  (per-clip and overall).
  4. Render a deck slide -> benchmark.png + table.md  (your data vs reference numbers).

Judge backend = ANY OpenAI-compatible /v1/chat/completions endpoint:
  * Local vLLM (recommended for batch/scale, no per-call cost):
      pip install vllm
      vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
      python run_eval.py --clips-dir ./clips \
          --base-url http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct
  * HF Inference Providers router (uses your HF token as the key; good for a quick try):
      python run_eval.py --clips-dir ./clips \
          --base-url https://router.huggingface.co/v1 \
          --api-key $HF_TOKEN --model Qwen/Qwen3-VL-8B-Instruct

IMPORTANT (benchmark validity): absolute percentages depend on the judge model.
Do NOT place these numbers next to Build AI's *published* (Gemini-judged) numbers and
call it apples-to-apples. To make the comparison self-consistent, re-judge reference
frames with THIS SAME script (point --clips-dir at reference frames) and pass the
resulting summary.json via --reference-json. See README.md.
"""

import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ----------------------------------------------------------------------------- prompts
# Verbatim from builddotai/Egocentric-100K-Evaluation (prompts/*.txt), with one
# explicit clarification added for factory data: gloved hands COUNT. Build AI's data
# is bare-handed; factory workers often wear gloves, and an un-clarified judge tends to
# under-count gloved hands -> you'd look worse than you are. Toggle with --no-glove-note.
GLOVE_NOTE = " A hand wearing a glove still counts as a visible hand."

HAND_COUNT_PROMPT = (
    "You are labeling an egocentric first-person image. Your task is to count how many "
    "camera-wearer's hands are visually present in the image: 0, 1, or 2.\n"
    "Rules:\n"
    "- Only count hands that are directly visible.\n"
    "- Do not infer hands that are outside the frame or potentially behind objects.\n"
    "- Ignore hands belonging to other people.\n"
    "- Any amount of visibility counts (even fingertips).{glove}\n"
    "- Return only one of: 0, 1, 2. No extra words."
)

ACTIVE_MANIP_PROMPT = (
    "You are labeling an egocentric first-person image. Your task is to determine whether "
    "the camera-wearer is actively manipulating an object at this exact moment.\n"
    "Definition: \"Active Manipulation\" means the wearer is visibly using their hands to "
    "work on, modify, assemble, process, or handle physical objects, materials, or "
    "components in pursuit of a specific goal.\n"
    "Rules:\n"
    "- Do not infer actions that are not visible in the frame.\n"
    "- If the action is ambiguous or not clearly happening, respond \"no\".\n"
    "- Ignore objects held by other people.\n"
    "- Respond only with: \"yes\" or \"no\"."
)

HAND_COUNT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "hand_count",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"hand_count": {"type": "integer", "enum": [0, 1, 2]}},
            "required": ["hand_count"],
        },
    },
}
ACTIVE_MANIP_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "active_manipulation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
            "required": ["answer"],
        },
    },
}

# Build AI's PUBLISHED numbers (judged by gemini-2.5-flash, 10k frames each).
# Source: huggingface.co/datasets/builddotai/Egocentric-100K-Evaluation (README).
# Shown for orientation ONLY; not a same-judge comparison unless you rescore (README).
REFERENCE_PUBLISHED = {
    "_note": "Build AI PUBLISHED numbers (gemini-2.5-flash judge). NOT same-judge as your run.",
    "Egocentric-100K (Build AI)": {"hands_1plus": 96.95, "hands_2": 79.05, "active": 92.76},
    "Ego4D":                      {"hands_1plus": 67.33, "hands_2": 36.95, "active": 50.07},
    "EPIC-KITCHENS":              {"hands_1plus": 90.37, "hands_2": 61.05, "active": 85.04},
}

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_TLS = threading.local()


# ----------------------------------------------------------------------------- ffmpeg
def ffmpeg_exe() -> str:
    """Prefer a system ffmpeg; fall back to the pip 'imageio-ffmpeg' static binary."""
    from shutil import which
    exe = which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("No ffmpeg found. Install one:  pip install imageio-ffmpeg")


def probe_duration(ff: str, clip: Path) -> float:
    """Seconds. Parsed from ffmpeg stderr 'Duration: HH:MM:SS.xx' (no ffprobe needed)."""
    out = subprocess.run([ff, "-hide_banner", "-i", str(clip)],
                         capture_output=True, text=True).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", out)
    if not m:
        return 0.0
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def extract_frame(ff: str, clip: Path, t: float, dst: Path, img_size: int) -> bool:
    if dst.exists():
        return True
    r = subprocess.run(
        [ff, "-y", "-nostdin", "-ss", f"{t:.3f}", "-i", str(clip), "-frames:v", "1",
         "-vf", f"scale={img_size}:-1", "-loglevel", "error", str(dst)],
        capture_output=True, text=True)
    return dst.exists() and r.returncode == 0


def plan_samples(ff: str, clips, n_frames: int, fps: float, seed: int):
    """Allocate ~n_frames across clips proportional to duration; evenly spaced timestamps."""
    durs = [(c, probe_duration(ff, c)) for c in clips]
    durs = [(c, d) for c, d in durs if d > 0.0]
    if not durs:
        sys.exit("No decodable clips found.")
    total = sum(d for _, d in durs)
    rng = random.Random(seed)
    plan = []  # (clip, t_seconds)
    for c, d in durs:
        cap = max(1, int(round(d * fps)))                       # frames available at --fps
        want = max(1, int(round(n_frames * d / total)))         # fair share by duration
        k = min(cap, want)
        if k == 1:
            ts = [d / 2.0]
        else:
            step = d / (k + 1)
            ts = [step * (i + 1) for i in range(k)]
        plan += [(c, t) for t in ts]
    rng.shuffle(plan)
    return plan[:n_frames]


# ----------------------------------------------------------------------------- judge
def session() -> requests.Session:
    s = getattr(_TLS, "s", None)
    if s is None:
        s = _TLS.s = requests.Session()
    return s


def judge_frame(args, img_path: Path, prompt: str, schema: dict) -> str:
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    payload = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "response_format": schema,
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    last = ""
    for attempt in range(args.retries + 1):
        try:
            r = session().post(f"{args.base_url.rstrip('/')}/chat/completions",
                               json=payload, headers=headers, timeout=args.timeout)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last = f"HTTP {r.status_code}: {r.text[:160]}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        # backoff
        import time
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"judge failed after {args.retries+1} tries: {last}")


def parse_hand(text: str):
    try:
        return int(json.loads(text)["hand_count"])
    except Exception:
        m = re.search(r"[012]", text or "")
        return int(m.group()) if m else None


def parse_active(text: str):
    try:
        return json.loads(text)["answer"].strip().lower()
    except Exception:
        t = (text or "").lower()
        return "yes" if "yes" in t else ("no" if "no" in t else None)


# ----------------------------------------------------------------------------- cache
def db_open(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("CREATE TABLE IF NOT EXISTS r "
                "(fid TEXT, prompt TEXT, val TEXT, PRIMARY KEY(fid,prompt))")
    con.commit()
    return con


def cache_get(con, lock, fid, prompt):
    with lock:
        row = con.execute("SELECT val FROM r WHERE fid=? AND prompt=?", (fid, prompt)).fetchone()
    return row[0] if row else None


def cache_put(con, lock, fid, prompt, val):
    with lock:
        con.execute("INSERT OR REPLACE INTO r VALUES (?,?,?)", (fid, prompt, val))
        con.commit()


# ----------------------------------------------------------------------------- report
def aggregate(rows):
    """rows: list of dicts with hand(int|None), active(str|None)."""
    hands = [r["hand"] for r in rows if r["hand"] is not None]
    acts = [r["active"] for r in rows if r["active"] is not None]
    n = len(hands) or 1
    out = {
        "frames_scored": len(rows),
        "hands_n": len(hands),
        "hands_0_pct": round(100 * sum(h == 0 for h in hands) / n, 2),
        "hands_1plus_pct": round(100 * sum(h >= 1 for h in hands) / n, 2),
        "hands_2_pct": round(100 * sum(h == 2 for h in hands) / n, 2),
        "active_n": len(acts),
        "active_pct": round(100 * sum(a == "yes" for a in acts) / (len(acts) or 1), 2),
    }
    return out


def render_chart(your, reference, out_png, your_label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[chart] matplotlib not installed; skipping benchmark.png")
        return
    metrics = [("hands_1plus", "1+ hands"), ("hands_2", "Both hands"), ("active", "Active manip.")]
    series = {your_label: {"hands_1plus": your["hands_1plus_pct"],
                           "hands_2": your["hands_2_pct"],
                           "active": your["active_pct"]}}
    for k, v in reference.items():
        if k.startswith("_"):
            continue
        series[k] = v
    labels = [m[1] for m in metrics]
    x = range(len(labels))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, vals) in enumerate(series.items()):
        bars = ax.bar([xx + i * width for xx in x],
                      [vals[m[0]] for m in metrics], width, label=name)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks([xx + width * (len(series) - 1) / 2 for xx in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of frames")
    ax.set_ylim(0, 105)
    ax.set_title("Egocentric data quality (Qwen3-VL-8B judge)")
    ax.legend(fontsize=8)
    if any(k.startswith("Egocentric-100K") and reference is REFERENCE_PUBLISHED for k in reference):
        fig.text(0.5, 0.01,
                 "Reference bars = Build AI PUBLISHED (gemini judge) - orientation only, "
                 "not same-judge. Rescore for a valid comparison.",
                 ha="center", fontsize=7, color="#888")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[chart] wrote {out_png}")


def render_table_md(your, reference, out_md, your_label):
    lines = ["| Dataset | 1+ hands | Both hands | Active manip. |",
             "|---|---|---|---|",
             f"| **{your_label}** | **{your['hands_1plus_pct']}%** | "
             f"**{your['hands_2_pct']}%** | **{your['active_pct']}%** |"]
    for k, v in reference.items():
        if k.startswith("_"):
            continue
        lines.append(f"| {k} | {v['hands_1plus']}% | {v['hands_2']}% | {v['active']}% |")
    if reference is REFERENCE_PUBLISHED:
        lines.append("")
        lines.append("_Reference rows = Build AI published (gemini-2.5-flash judge); "
                     "shown for orientation. For a same-judge comparison, rescore reference "
                     "frames with this script and pass --reference-json._")
    out_md.write_text("\n".join(lines))
    print(f"[table] wrote {out_md}\n")
    print("\n".join(lines))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Egocentric data-quality eval (Qwen3-VL-8B judge)")
    ap.add_argument("--clips-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("eval_out"))
    ap.add_argument("--n-frames", type=int, default=10000)
    ap.add_argument("--fps", type=float, default=1.0, help="sampling fps within each clip")
    ap.add_argument("--img-size", type=int, default=768, help="long-side px sent to the judge")
    ap.add_argument("--seed", type=int, default=0)
    # judge endpoint (OpenAI-compatible)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--no-glove-note", action="store_true",
                    help="drop the 'gloved hands count' clarification (match Build AI exactly)")
    ap.add_argument("--your-label", default="Our data (factory)")
    ap.add_argument("--reference-json", type=Path,
                    help="summary.json from rescoring reference frames with THIS judge "
                         "(makes the comparison same-judge). Default: Build AI published numbers.")
    ap.add_argument("--limit-clips", type=int, default=0, help="debug: cap number of clips")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    ff = ffmpeg_exe()

    clips = sorted(p for p in args.clips_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if args.limit_clips:
        clips = clips[:args.limit_clips]
    if not clips:
        sys.exit(f"No videos under {args.clips_dir} (looked for {sorted(VIDEO_EXTS)})")
    print(f"[plan] {len(clips)} clips -> sampling ~{args.n_frames} frames @ {args.fps} fps")

    glove = "" if args.no_glove_note else GLOVE_NOTE
    hand_prompt = HAND_COUNT_PROMPT.format(glove=glove)

    plan = plan_samples(ff, clips, args.n_frames, args.fps, args.seed)
    print(f"[plan] {len(plan)} (clip, timestamp) samples")

    con = db_open(args.out_dir / "cache.sqlite")
    lock = threading.Lock()

    def work(item):
        clip, t = item
        fid = hashlib.md5(f"{clip}|{t:.3f}|{args.img_size}".encode()).hexdigest()
        jpg = frames_dir / f"{fid}.jpg"
        if not extract_frame(ff, clip, t, jpg, args.img_size):
            return None
        # hand_count
        hv = cache_get(con, lock, fid, "hand")
        if hv is None:
            hv = judge_frame(args, jpg, hand_prompt, HAND_COUNT_SCHEMA)
            cache_put(con, lock, fid, "hand", hv)
        # active_manipulation
        av = cache_get(con, lock, fid, "active")
        if av is None:
            av = judge_frame(args, jpg, ACTIVE_MANIP_PROMPT, ACTIVE_MANIP_SCHEMA)
            cache_put(con, lock, fid, "active", av)
        return {"clip": str(clip), "t": round(t, 3), "fid": fid,
                "hand": parse_hand(hv), "active": parse_active(av)}

    rows, errors = [], 0
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(plan), desc="judging")
    except Exception:
        pbar = None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, it) for it in plan]
        for f in as_completed(futs):
            try:
                r = f.result()
                if r:
                    rows.append(r)
            except Exception as e:  # noqa: BLE001
                errors += 1
                if errors <= 5:
                    print(f"[warn] {e}", file=sys.stderr)
            if pbar:
                pbar.update(1)
    if pbar:
        pbar.close()

    if not rows:
        sys.exit("No frames scored - check the judge endpoint (--base-url/--model/--api-key).")

    # per-frame csv
    csv_path = args.out_dir / "frames.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip", "t", "fid", "hand", "active"])
        w.writeheader()
        w.writerows(rows)

    overall = aggregate(rows)
    overall["errors"] = errors
    overall["judge_model"] = args.model
    overall["glove_note"] = not args.no_glove_note
    (args.out_dir / "summary.json").write_text(json.dumps(overall, indent=2))
    print(f"\n[summary] {json.dumps(overall, indent=2)}")

    reference = REFERENCE_PUBLISHED
    if args.reference_json and args.reference_json.exists():
        ref = json.loads(args.reference_json.read_text())
        reference = {"Reference (same judge)": {
            "hands_1plus": ref["hands_1plus_pct"], "hands_2": ref["hands_2_pct"],
            "active": ref["active_pct"]}}
        print("[ref] using same-judge reference from", args.reference_json)
    else:
        print("[ref] WARNING: comparing against Build AI PUBLISHED (gemini) numbers - "
              "orientation only, NOT same-judge. See README to rescore.")

    render_table_md(overall, reference, args.out_dir / "table.md", args.your_label)
    render_chart(overall, reference, args.out_dir / "benchmark.png", args.your_label)
    print(f"\n[done] outputs in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
