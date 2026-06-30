# CLAUDE.md

EMBR egocentric-data pipeline. EMBR is a robotics-data company that turns first-person (egocentric) factory-worker video into clean, richly-labeled datasets good enough to train robot manipulation policies. This repo takes raw iPhone clips of a car-interior-trim assembly line, preprocesses them into VLM-ready frames, runs a hand-visibility / active-manipulation quality benchmark, and renders MediaPipe 3D hand-pose + VLM-caption overlays.

Two product paths frame everything here:
- **PATH 2 (PRIMARY)** — the labeled dataset is the product you sell.
- **PATH 1 (VALIDATION)** — a downstream retarget + small policy-training run that proves the labels are robot-grade. This is the credible demo metric (mirrors Scale AI's "validate labels by fine-tuning a model on them").

---

## Repository & layout

- **Repo:** https://github.com/EmbrData/embr-pipeline (private, EmbrData org, default branch `main`)
- **Description:** Egocentric factory-worker video → robot-grade labeled dataset: preprocess, Qwen3-VL quality benchmark, MediaPipe 3D hand-pose + VLM-caption overlays.

### Committed file tree
```
.gitignore
README.md
requirements.txt
setup.sh
preprocess.py                      # stage 1: raw video -> VLM-ready frames
benchmark/README.md                # judge-backend + cost + same-judge guidance
benchmark/run_eval.py              # stage 2: Qwen3-VL quality benchmark
overlay/overlay.py                 # stage 3a: single hero still overlay
overlay/video_overlay.py           # stage 3b: per-frame clip overlay
overlay/montage.py                 # stage 3c: RAW | AUTO-LABELED side-by-side montage
examples/hero_frame.jpg            # default input for overlay.py
models/.gitkeep                    # placeholder; the .task model is downloaded, not committed
results/benchmark_allclips.png     # per-clip bar chart across the 5 EMBR clips vs Build AI
results/benchmark_table.png        # rendered results table
results/hero_overlay.png           # hero still with hand-landmark overlay
results/hero_overlay_5s.mp4        # 5s hero clip overlay
results/embr_montage_15s.mp4       # 15s RAW|labeled montage
```

### Git-ignored (never committed)
- **Raw / proprietary video:** `clips/`, `*.MOV`, `*.mov`, `*.zip`
- **Generated intermediates:** `processed/`, `seg/`, `seg[0-9]*/`, `seg_out/`, `montage/`, `sample_frames/`, `frames/`, `bench/frames/`, `bench/ref/`, `eval_out/`, `cache.sqlite`, `*_scan.jpg`
- **Model weights:** `models/*.task` (downloaded by `setup.sh`; only `models/.gitkeep` is tracked)
- **Python / OS:** `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `.DS_Store`, `.env`

### Where things actually live on the dev Mac
- **Local working dir:** `/Users/benzh1/Documents/EMBR-pipeline` — the ACTUAL 5 source clips, all intermediate frame dirs (`processed/`, `seg*/`, `bench/`, `hero/`, `montage/`) and rendered outputs. NOT in git.
- The repo tree above is the shippable, organized mirror; the local dir is where the real data and demo intermediates sit.
- A secondary copy of the eval script source also exists at `/Users/benzh1/egocentric-eval/`.
- **Cleanup awareness:** three `drive-download-*.zip` archives (~5.07 GB total) at the local repo root mirror the clip sizes — they are the original Google Drive download bundles of the source clips, redundant once `clips/` is populated, and the single largest reclaimable space on disk.

---

## Quickstart / environment setup

### Hard constraint: Python 3.9–3.12 + MediaPipe
MediaPipe ships **no 3.13/3.14 wheels**, so `mediapipe>=0.10.18` will not install on a newer interpreter. This is called out in both `requirements.txt` (header comment) and `setup.sh` (step 1 message). MediaPipe 0.10.35's wheel also ships **only the Tasks API** — `mp.solutions` does NOT exist; you must use `mediapipe.tasks.python.vision.HandLandmarker` with the downloaded `hand_landmarker.task` bundle.

### Clean-machine setup (recommended for a new engineer)
```bash
# 1. Clone (private repo — needs gh auth or SSH access to the EmbrData org)
git clone https://github.com/EmbrData/embr-pipeline.git
cd embr-pipeline

# 2. Single Python 3.9–3.12 venv (3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install deps + download the MediaPipe hand_landmarker model
./setup.sh
#  (equivalently: pip install -r requirements.txt, then curl the model into
#   models/hand_landmarker.task)

# 4. Run the stages (raw clips go in ./clips/, git-ignored)
python preprocess.py
python overlay/overlay.py
python overlay/video_overlay.py
python overlay/montage.py
python benchmark/run_eval.py        # see benchmark/README.md
```

`setup.sh` pulls the model from `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` → `models/hand_landmarker.task` (~7.8 MB).

**Key deps (`requirements.txt`):** `mediapipe>=0.10.18`, `opencv-python>=4.9`, `pillow>=10.0`, `numpy>=1.24`, `imageio-ffmpeg>=0.5` (bundled ffmpeg), `requests>=2.31` (Qwen3-VL judge HTTP client), `matplotlib>=3.8` (charts), `tqdm>=4.66`.

### The actual two-venv reality on this dev Mac
This Mac has only **Python 3.14 system-wide** (no MediaPipe wheels) and **no system ffmpeg** (`which ffmpeg` → not found). So two venvs were created and the work is split between them:

| venv | Python | Provides | Verified version |
|------|--------|----------|------------------|
| `/tmp/ffv` | 3.14.6 | `imageio-ffmpeg` only — a static ffmpeg binary | imageio_ffmpeg 0.6.0 (bundles `ffmpeg-macos-aarch64-v7.1`) |
| `/tmp/mpv` | 3.11.15 (`brew install python@3.11`) | `mediapipe`, `opencv-python`, `pillow`, `numpy` | mediapipe 0.10.35, cv2 4.13.0 |

**Why split:** `imageio_ffmpeg` is pure glue with broad wheel support, so it can run on the newest 3.14 and supply ffmpeg; MediaPipe cannot, so all vision work is pinned to the 3.11 venv. Run all MediaPipe scripts with `/tmp/mpv/bin/python`.

Get the bundled ffmpeg path:
```bash
/tmp/ffv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
# -> /private/tmp/ffv/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1
```

The on-disk model used by the overlay scripts here is `/Users/benzh1/Documents/EMBR-pipeline/hero/hand_landmarker.task` (7,819,105 bytes). Load it via `mediapipe.tasks.python.vision.HandLandmarker`, never `mp.solutions.hands`.

---

## The pipeline, stage by stage

### Stage 1 — `preprocess.py` (egocentric clip → VLM-ready frames)
Normalizes raw 4K HDR iPhone capture into JPGs a VLM can consume, handling HDR tone-mapping and rotation per clip.

```bash
python preprocess.py --clips-dir clips --out-dir processed --frames-per-clip 48
```

CLI flags (exact defaults):

| Flag | Default | Meaning |
|---|---|---|
| `--clips-dir` | required | Input dir of source clips (**non-recursive** `iterdir`). |
| `--out-dir` | `processed` | Output dir for JPGs + manifest. |
| `--frames-per-clip` | `48` | Evenly-spaced frames/clip. Equal n = fair per-clip compare. |
| `--fps` | `0.0` | Override: when `>0`, `n = max(1, int(dur*fps))` and overrides `--frames-per-clip`. |
| `--img-size` | `768` | Long-side downscale target (`scale={img_size}:-1`) to control VLM token cost. |

What it does: resolves ffmpeg (system, else `imageio-ffmpeg` static binary); collects clips by suffix (`.mov .mp4 .mkv .m4v .avi .webm`); probes each via `ffmpeg -i` stderr (`Duration:` regex for seconds; scans the `Video:` line for HDR markers `arib-std-b67`, `smpte2084`, `bt2020` → `is_hdr`); if HDR, prepends the `TONEMAP` filter (zscale bt2020→linear→bt709 + `tonemap=hable:desat=0` → yuv420p) before `scale`; extracts each frame via **fast-seek** (`-ss` before `-i`, `-frames:v 1`, `-qscale:v 3`) at evenly-spaced mid-bin timestamps `t = dur*(i+0.5)/n`, one ffmpeg call per frame.

**Produces** (in `--out-dir`): `<clipstem>_001.jpg … _NNN.jpg` (spaces→`_`, 3-digit index); `manifest.json` (`{name, source, duration_s, hdr, prefix, count, frames[]}`); and a printed **GROUPS spec** (`{ name, prefix, count, hdr, bucket: 'user' }`) to paste into the benchmark workflow.

### Stage 2 — `benchmark/run_eval.py` (egocentric data-quality eval, open VLM judge)
Reproduces Build AI's `Egocentric-100K` quality benchmark (**hand visibility** + **active manipulation**) on your clips using `Qwen3-VL-8B-Instruct` as judge via any OpenAI-compatible endpoint. No Gemini/Google key. See `benchmark/README.md`.

```bash
# HF Inference Providers router (Novita) — HF fine-grained token IS the api key
export HF_TOKEN=hf_...
python benchmark/run_eval.py \
  --clips-dir processed \
  --base-url https://router.huggingface.co/v1 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --no-glove-note
```

Key CLI flags (exact defaults):

| Flag | Default | Meaning |
|---|---|---|
| `--clips-dir` | required | Input dir, searched **recursively** (`rglob`). |
| `--out-dir` | `eval_out` | Output dir. |
| `--n-frames` | `10000` | Target total frames across all clips. |
| `--fps` | `1.0` | Sampling fps **within** each clip (caps per-clip frames). |
| `--img-size` | `768` | Long-side px sent to judge. |
| `--seed` | `0` | RNG seed for the sample-plan shuffle. |
| `--base-url` | `http://localhost:8000/v1` | OpenAI-compatible endpoint base. |
| `--model` | `Qwen/Qwen3-VL-8B-Instruct` | Judge model id. |
| `--api-key` | `$OPENAI_API_KEY` or `$HF_TOKEN` | Bearer key; may be `None`. |
| `--workers` | `8` | ThreadPoolExecutor size. |
| `--retries` | `4` | Retries/call (5 total tries, exp backoff `min(2**attempt,20)`). |
| `--timeout` | `120` | Per-request seconds. |
| `--no-glove-note` | off | Drops the "gloved hands count" clarification to match Build AI's exact wording. |
| `--your-label` | `Our data (factory)` | Label for your row in table/chart. |
| `--reference-json` | `None` | A prior `summary.json` to use as a **same-judge** reference. |
| `--limit-clips` | `0` | Debug: cap number of clips (`0` = no cap). |

What it does: plans samples proportional to clip duration (capped by `--fps`), seeded-shuffled, truncated to `--n-frames`; opens a resumable `cache.sqlite`; for each `(clip, t)` derives `fid = md5(f"{clip}|{t:.3f}|{img_size}")`, extracts a frame to `frames/<fid>.jpg`, and judges twice — `hand_count` ({0,1,2}) and `active_manipulation` ({yes,no}) — each `temperature=0, max_tokens=16`, structured `response_format` json_schema (`strict:True`), cache-checked before / cache-put after.

**Produces** (in `--out-dir`): `frames/<fid>.jpg`; `cache.sqlite`; `frames.csv` (`clip, t, fid, hand, active`); `summary.json` (`frames_scored, hands_n, hands_0_pct, hands_1plus_pct, hands_2_pct, active_n, active_pct, errors, judge_model, glove_note`); `table.md`; `benchmark.png` (skipped if matplotlib missing).

### Stage 3 — overlays (`overlay/`)
Shared chassis: real MediaPipe Hand Landmarker (21 3D keypoints/hand, Apache-2.0), PIL-drawn overlay (lime/cyan skeleton, white joints, bbox, per-hand chip, "EMBR auto-labeling" badge, caption bar). All require `models/hand_landmarker.task`. **Captions/objects are authored VLM stand-ins (static literals), not live model output** — only the hand pose is real.

```bash
python overlay/overlay.py          # single hero still: examples/hero_frame.jpg -> results/hero_overlay.png
python overlay/video_overlay.py    # per-frame clip: seg/*.jpg -> seg_out/NNNN.png (encode to mp4 via ffmpeg)
python overlay/montage.py          # 15s RAW|EMBR AUTO-LABELED montage: montage/NNNNN.png
```

- **`overlay.py`** — `RunningMode.IMAGE`, `num_hands=2`, `min_hand_detection_confidence=0.3`. Coloring is **index-based** (`COLORS[hi % 2]`, lime/cyan) — this is the one script NOT colored by handedness. Config consts at top: `IN`, `OUT`, `MODEL`. Hardcoded `ACTION`/`META`/`OBJECTS` stand-ins.
- **`video_overlay.py`** — `RunningMode.VIDEO`, all three confidences 0.3, `detect_for_video(..., int(i*1000/FPS))` at `FPS=30`. Coloring is **handedness-based** (`HCOL` keyed on `Left`/`Right`). Reads `SRC="seg"`, writes `DST="seg_out"`.
- **`montage.py`** — `RunningMode.VIDEO`, `CONF=0.3` on all three confidences, handedness coloring, `FPS=30`. `WINDOWS` is a list of 3 `{dir, action, objects}` dicts (`seg`, `seg7`, `seg5`); geometry probed from `WINDOWS[0]["dir"]`'s first JPG. Writes `montage/NNNNN.png` (global sequential index) and prints per-window frame count + two-hand %.

**These are the CLEAN settings** (VIDEO mode, confidence 0.3, handedness coloring, no post-hoc tricks). Confirmed absent: mirror-merge, skin-filter, temporal-fill (see Gotchas).

---

## Design plan (9-stage architecture; path-2 product vs path-1 validation)

The pipeline architecture follows the human-video→robot-manipulation survey (arXiv 2606.00054), which structures the problem in 3 stages: (a) structure video into episodes; (b) ground into robot-executable actions despite the embodiment gap; (c) deployment-predictive eval. Of the 4 video→VLA supervision technique classes (latent actions, world models, explicit 2D, explicit 3D), hand-centric egocentric data sits in the **explicit-3D (3D hand pose)** class — which is why the pipeline is built around hand pose.

The full conceptual pipeline runs: capture → preprocess (HDR/rotate/sample) → quality benchmark (QA gate) → 3D hand-pose extraction → VLM caption/object labeling → episode structuring → action grounding/retargeting → policy-training validation → dataset export. The repo today implements the front half (preprocess, benchmark, hand-pose + caption overlay); the back half (retarget, training validation, export) is the path-1 roadmap.

- **PATH 2 (product, what ships now):** MediaPipe 3D hand pose + VLM captions → labeled egocentric dataset, exported in a robot-standard format. The overlay code is the shippable labeling engine.
- **PATH 1 (validation, the credible metric):** retarget MANO/wrist params to a robot action space (IK), run a small diffusion-policy / VLA fine-tune, and report success rate as the dataset-quality number. Canonical recipe = EgoVLA.

---

## Research context

Pulled from a fact-checked deep-research pass. Vendor/competitor claims are vendor-reported, not independently verified. Curation findings are lab-validated, not yet validated on egocentric factory video.

**Architecture & canonical recipe:**
- Survey **arXiv 2606.00054** ("From Human Videos to Robot Manipulation") — the 3-stage frame + 4 technique classes above.
- **EgoVLA (arXiv 2507.12440)** — the canonical recipe: pretrain a VLA on human video predicting wrist + MANO hand params (unified human/robot action space), IK + retarget to the robot, fine-tune on a few robot demos. **CAVEAT: zero-shot robot transfer = 0%; fine-tuning is required.**
- **EgoDex (arXiv 2505.11709, Apple)** — 829h egocentric + 3D hand pose, but captured with a Vision Pro rig (not post-hoc). Use as a schema reference, not a method to replicate.
- Others: Do-as-I-Do (2606.19333), MotionTrans (2509.17759, open-sourced, names the 3 stages), In-N-On (2511.15704), EgoMimic, EgoZero (2505.20290).

**Curation beats scale (strongly corroborated):**
- What-Matters (2506.13536): retrieval beats full-DROID by up to 70%; camera-pose alignment +25%; object texture minimal.
- CUPID (2506.19121): <33% of data = SOTA via influence functions.
- SCIZOR (2505.22626): +15.4%, self-supervised. DemInf (2502.08623), DataMIL (2505.09603), Data Scaling Laws (2410.18647): environment/object diversity > demo count.
- **CAVEAT:** all lab-validated (RoboMimic / ALOHA / Franka / DROID), NOT egocentric factory video — transfer is unvalidated. Diversity > volume is the working thesis, not proof.

**Competitors (vendor-reported):**
- **Build AI** = closest analog and the benchmark we reproduce. Claims 100k hrs factory-worker footage; open HF dataset `builddotai/Egocentric-100K` (GATED; WebDataset of `{mp4, json}`; 238 factories × 14,228 workers × ~30k shards; per-worker camera intrinsics). Public eval (ungated `builddotai/Egocentric-100K-Evaluation`) scores hand-visibility + active-manipulation with **gemini-2.5-flash: 96.95% (1+ hand) / 79.05% (both) / 92.76% (active)**. Reference subset also at `Voxel51/Egocentric_10K_subset` (ungated).
- **Claru** = pure-play egocentric, delivers RLDS/LeRobot/HDF5, request-only, NO factory domain.
- **Scale AI Physical AI** = data engine + teleoperation; validates labels by fine-tuning (the path-1 model).
- **Encord / iMerit** = annotation platforms. **Mercor / Datacurve are NOT robotics data** (talent / code-RLHF marketplaces) — do not mis-scope them.
- Adopt **LeRobot / RLDS** as the dataset export format.

**Hand-pose model decision (MediaPipe vs HaMeR):**
- **HaMeR** (github.com/geopavlakos/hamer, arXiv 2312.05251): ViT-H → MANO 3D mesh (778 verts); robust on egocentric / occlusion / gloves / tool-grip. BUT no hosted inference (self-host only, CUDA GPU + ViTDet/ViTPose detector), and **MANO is NON-COMMERCIAL** (registration + separate Max Planck commercial license); **WiLoR weights are CC-BY-NC**. → licensing blocker for shipping HaMeR-derived labels.
- **MediaPipe Hand Landmarker**: 21 keypoints/hand (+ optional metric world landmarks), **Apache-2.0**, CPU/Mac, real-time. Weaker on closed-fist tool-grips. **This is the workhorse we ship now** (path 2 + demos). Swap in HaMeR for a premium 3D-mesh tier once MANO is licensed and GPUs are available — the overlay code is structured for a drop-in engine swap.

**VLM judge / captioner — Qwen3-VL-8B-Instruct:**
- Served via **HF Inference Providers** (provider Novita) on the OpenAI-compatible router (`https://router.huggingface.co/v1`; an HF fine-grained token with "Make calls to Inference Providers" permission IS the API key), or self-host **vLLM** (official FP8 ~16 GB VRAM; BF16 ~24 GB).
- A 10K-frame eval costs **~$1.7** (~1.2K in + 150 out tokens/frame; $0.08/M-in + $0.50/M-out). HF monthly credits: Free $0.10, PRO $2.00 → fits PRO. Self-hosted FP8 undercuts serverless (~$171/1M) at batch scale.
- **Gemini 2.5 Flash was considered only for parity with Build AI's published numbers and DROPPED** ("not worth the Google key"). Use Qwen3-VL as the judge and re-judge any reference set with the SAME judge for a valid comparison.

---

## Data findings & benchmark methodology

**Method:** reproduce Build AI's eval. Sample frames; judge each with two structured-output prompts — `hand_count` {0,1,2} and `active_manipulation` {yes,no} (verbatim Build AI prompts, plus a "gloved hands count" clarification; drop it with `--no-glove-note` to match Build AI exactly).

**SAME-JUDGE RULE (the big one):** absolute percentages depend on the judge model. Never compare your-judge numbers directly to Build AI's published gemini numbers — re-judge any reference set with the SAME judge (via `--reference-json`) before comparing. The benchmark on this Mac used **Claude (Opus) as a STAND-IN judge** because the machine has no GPU/token; **production uses Qwen3-VL via `run_eval.py`**. The published `REFERENCE_PUBLISHED` numbers in the script (Egocentric-100K 96.95/79.05/92.76, Ego4D, EPIC-KITCHENS) are orientation only; the script prints a `[ref] WARNING` and footnotes the chart when it falls back to them.

### Per-clip ranking (48 frames/clip, Claude judge)
| Clip | ≥1 hand % | Both-hands % | Active-manip % |
|---|---|---|---|
| IMG_1534 (1×, camera aimed at the work) | 100 | 72.9 | 89.6 |
| IMG_1539 (0.5× ultrawide) | 100 | 70.8 | 87.5 |
| IMG_1533 (chest-down) | 81.3 | 43.8 | 68.8 |
| IMG_1532 (mixed) | 77.1 | 27.1 | 62.5 |
| IMG_1531 (chest-down) | 68.8 | 27.1 | 54.2 |
| **EMBR all-5 (pooled)** | **85.4** | **48.3** | **72.5** |

### EMBR vs Build AI (same judge) vs Build AI published Gemini
| Source | ≥1 hand % | Both-hands % | Active-manip % |
|---|---|---|---|
| EMBR best clip (IMG_1534) | 100 | 72.9 | 89.6 |
| EMBR all-5 pooled | 85.4 | 48.3 | 72.5 |
| Build AI reference (44 frames, our judge) | 95.5 | 72.7 / 75.0 | 75.0 / 77.3 |
| Build AI published Gemini | 96.95 | 79.05 | 92.76 |

**Takeaway:** EMBR's best clip (IMG_1534) essentially matches Build AI's reference on both-hands (72.9 vs ~72.7) and beats it on active-manipulation (89.6 vs 75.0), approaching Build AI's published Gemini active number (92.76). Pooled across all 5 clips EMBR trails, dragged down by the two chest-down clips (IMG_1531/1533) where framing cuts off hands.

**Note on the two JSON files:** the two source files disagree on Build-AI-reference both-hands/active (72.7/75.0 vs 75.0/77.3) and on per-clip framing. `ranking_summary.json` is the 48-frames/clip ranking run; `benchmark_summary.json` is a separate 154-frame run (EMBR all = 110 frames: 71.8 / 21.8 / 42.7, 28.2% zero-hands; IMG_1531 = 65 frames 64.6/15.4/32.3; IMG_1532 = 45 frames 82.2/31.1/57.8). Both are "same judge, our side" measurements; the published Gemini numbers are Build AI's own, not re-judged. Both are reported above rather than reconciled. Source: `bench/ranking_summary.json`, `bench/benchmark_summary.json`.

### KEY INSIGHT — quality is driven by CAMERA FRAMING, not the lens
The strongest clips (1534, 1539) keep the hands in frame; the chest-down clips cut hands off.
- **Capture SOP:** aim the camera at the work zone (like 1534), use the 1× lens, avoid pure chest-down framing.
- **Pitch guidance:** do NOT lead with raw hand-density (you'd lose to Build AI's larger corpus). Lead with the **TRANSFORMATION** (auto-labeling raw video into robot-ready labels) + **factory-domain ownership** + the **path-1 validation metric**. The benchmark is your internal QA instrument, not the headline.

---

## Gotchas / hard-won lessons

- **HDR tone-map + rotation are mandatory in preprocessing.** Clips are 4K (3840×2160) 10-bit HLG HDR (BT.2020 / `arib-std-b67`), portrait, rotated −90°. Frame extraction MUST tone-map HDR→SDR (zscale linear → hable tonemap → bt709) and de-rotate. HDR detection is **substring-based** on the `Video:` line — a mislabeled/odd container silently skips tone-mapping and frames "look wrong to the VLM." Rotation is handled by **ffmpeg autorotate, which is ON BY DEFAULT** — there is no flag and the script never passes `-noautorotate`; do NOT add it. `run_eval.py`'s extractor is plain `scale` only (no HDR/rotation handling) — feed it `preprocess.py` output or already-clean SDR/upright clips.
- **`--clips-dir` recursion differs:** `preprocess.py` is **non-recursive** (`iterdir`); `run_eval.py` **recurses** (`rglob`).
- **Fast-seek extraction:** `-ss` before `-i`, one frame per ffmpeg call — ~100× faster than the fps filter on long 4K HEVC, at the cost of many short subprocesses. Output count is re-globbed from disk, so a frame may be `< n` if an extraction fails silently (failures aren't surfaced beyond the `[ok] … N frames` line).
- **MediaPipe Tasks-API only:** `mp.solutions` does not exist in 0.10.35 — use `mediapipe.tasks.python.vision.HandLandmarker` + `hand_landmarker.task`.
- **The benchmark sqlite cache does NOT bust on prompt change.** `cache.sqlite` keys judgments by `(fid, prompt)` where `prompt` is the literal string `"hand"`/`"active"`, not the prompt body, and `fid` includes `img_size`. Toggling `--no-glove-note` reuses stale `hand` verdicts — **delete `cache.sqlite` (or the out-dir) when you change prompts/glove handling.** Changing `--img-size` *does* change `fid` → fresh frames + re-judge.
- **`--api-key` may silently be `None`** if neither `OPENAI_API_KEY` nor `HF_TOKEN` is set (no Authorization header) — fine for local vLLM, fails on the HF router. Glove note is ON by default and inflates gloved-hand counts vs Build AI; calibrate with a ~100-frame pilot before a 1M-frame run.
- **REVERTED post-hoc tricks — mirror-merge + skin-tone filter + temporal-fill.** These raised two-hand recall but introduced phantom hands (a skeleton on a screw bin) and stale held hands. They were removed; the overlay scripts are confirmed free of mirror-merge, skin-filter, and temporal-fill. **Lesson: prefer picking a well-framed window over post-hoc tricks.** To get a strong demo window, scan the clip for high two-hand-detection regions (IMG_1539 t≈33s = 96% two-hand) rather than forcing a weak one.
- **Coloring inconsistency:** `overlay.py` colors by detection index (`COLORS[hi % 2]`); both video scripts color by handedness (`HCOL[Left/Right]`). If "clean settings" means handedness-coloring everywhere, the hero still does not match — but it is cosmetic and detection settings are identical.
- **Next-stage HaMeR licensing trap:** MANO is non-commercial (registration-gated `MANO_RIGHT.pkl`), WiLoR weights are CC-BY-NC. Use HaMeR for internal validation only until a commercial hand-pose license is cleared.

---

## Claude Code / ultracode workflow guide

We used Claude Code's Workflow (multi-agent orchestration) tool throughout. Saved workflow scripts live under the Claude session's `workflows/scripts/` dir. The seven EMBR-relevant scripts:

| Workflow | Script path | Purpose |
|---|---|---|
| deep-research | `…/sharpe-agent-py/3d65ebda-…/workflows/scripts/deep-research-wf_1008b288-d51.js` | fan-out web search + adversarial 3-vote verify + cited synthesis (produced the research above) |
| competitor-teardown | `…/sharpe-agent-py/76c0bbf8-…/workflows/scripts/competitor-teardown-wf_7f574efb-831.js` | parallel fetch + verify of vendor pages |
| inference-hosting-verify | `…/-76c0bbf8-…-subagents-…-wf-7f574efb-831/…/workflows/scripts/inference-hosting-verify-wf_7e05661d-204.js` | verify HF / Qwen3-VL / HaMeR / Gemini hosting + cost facts (subagent of competitor-teardown) |
| embr-data-benchmark | `…/EMBR-pipeline-bench/76c0bbf8-…/workflows/scripts/embr-data-benchmark-wf_6abd0e00-7e9.js` | benchmark the pipeline over a data set (rooted in `EMBR-pipeline-bench`) |
| embr-allclips-benchmark | `…/EMBR-pipeline/76c0bbf8-…/workflows/scripts/embr-allclips-benchmark-wf_35af3d52-29f.js` | run the benchmark across all clips (rooted in `EMBR-pipeline`) |
| hero-frame-finder | `…/sharpe-agent-py/76c0bbf8-…/workflows/scripts/hero-frame-finder-wf_602a1ffc-325.js` | fan-out agents scoring frames for hero-suitability |
| embr-handoff-claudemd | `…/sharpe-agent-py/76c0bbf8-…/workflows/scripts/embr-handoff-claudemd-wf_5a6e855f-545.js` | generate this handoff / CLAUDE.md |

Full paths (the `…` prefix is `/Users/benzh1/.claude/projects`):
- `/Users/benzh1/.claude/projects/-Users-benzh1-Documents-sharpe-agent-py/3d65ebda-d92d-46c4-850f-b1b3b596d685/workflows/scripts/deep-research-wf_1008b288-d51.js`
- `/Users/benzh1/.claude/projects/-Users-benzh1-Documents-sharpe-agent-py/76c0bbf8-b94a-4e47-89e5-fc582ed28251/workflows/scripts/competitor-teardown-wf_7f574efb-831.js`
- `/Users/benzh1/.claude/projects/-Users-benzh1--claude-projects--Users-benzh1-Documents-sharpe-agent-py-76c0bbf8-b94a-4e47-89e5-fc582ed28251-subagents-workflows-wf-7f574efb-831/76c0bbf8-b94a-4e47-89e5-fc582ed28251/workflows/scripts/inference-hosting-verify-wf_7e05661d-204.js`
- `/Users/benzh1/.claude/projects/-Users-benzh1-Documents-EMBR-pipeline-bench/76c0bbf8-b94a-4e47-89e5-fc582ed28251/workflows/scripts/embr-data-benchmark-wf_6abd0e00-7e9.js`
- `/Users/benzh1/.claude/projects/-Users-benzh1-Documents-EMBR-pipeline/76c0bbf8-b94a-4e47-89e5-fc582ed28251/workflows/scripts/embr-allclips-benchmark-wf_35af3d52-29f.js`
- `/Users/benzh1/.claude/projects/-Users-benzh1-Documents-sharpe-agent-py/76c0bbf8-b94a-4e47-89e5-fc582ed28251/workflows/scripts/hero-frame-finder-wf_602a1ffc-325.js`
- `/Users/benzh1/.claude/projects/-Users-benzh1-Documents-sharpe-agent-py/76c0bbf8-b94a-4e47-89e5-fc582ed28251/workflows/scripts/embr-handoff-claudemd-wf_5a6e855f-545.js`

### Three workflow gotchas (will save you time)
1. **`args` was unreliable in this environment — HARDCODE config as consts in the script** instead of passing args. Edit the const block at the top of the saved `.js` and re-run.
2. **A background workflow can die silently if the session forks.** Resume with `Workflow({scriptPath, resumeFromRunId})` — cached agent results return instantly, so you don't re-pay for completed fan-out work.
3. **Default workflow subagents CAN Read local image files (vision).** This is what made the GPU-free judge fan-out possible: fan out subagents-as-VLM-judges, each Reads a batch of local frames and returns `hand_count` + `active_manipulation`. That is how the Claude-stand-in benchmark numbers above were produced without a GPU or API token. (Production should still use Qwen3-VL via `run_eval.py` for the same-judge comparison to hold.)

---

## Open items / next steps

- **Clear a commercial MANO license (Max Planck)** before shipping any HaMeR-derived labels.
- **Wire overlay captions to a live Qwen3-VL endpoint** (currently authored stand-in text in `overlay/*.py`).
- **Build the HaMeR integration stub (GPU)** for path-1 retargeting (MANO → robot action space). The overlay engine is structured for a drop-in swap.
- **Write a one-page capture SOP** (camera at the work zone, 1× lens, avoid chest-down) → re-shoot a standardized session → re-run the benchmark to show the lift.
- **Adopt LeRobot / RLDS** as the export format.
- **Run the FULL Qwen3-VL benchmark** (not the Claude stand-in) on the full footage; collect more data across more workers/stations (diversity > volume).
- **Stand up the path-1 validation harness:** retarget + a small diffusion-policy / VLA training run whose success rate becomes the dataset-quality number for the pitch.