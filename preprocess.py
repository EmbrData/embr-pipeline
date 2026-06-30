#!/usr/bin/env python3
"""
preprocess.py - normalize egocentric clips into VLM-ready frames.

Per clip, auto-handles the EMBR capture quirks:
  * HDR -> SDR : 10-bit HLG/PQ (bt2020) tone-mapped to bt709 (else frames look wrong to a VLM)
  * rotation  : iPhone -90 deg display-matrix applied (ffmpeg autorotate, on by default)
  * downscale : long side -> --img-size (default 768) to control VLM token cost
  * sampling  : --frames-per-clip evenly-spaced frames (equal n => fair per-clip benchmark),
                or --fps for a fixed cadence.

Outledger: <out-dir>/<clipstem>_001.jpg ... + manifest.json + a printed GROUPS spec
to paste into the benchmark workflow.

Uses the pip 'imageio-ffmpeg' static binary if no system ffmpeg is present.
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

VIDEO_EXTS = {".mov", ".mp4", ".mkv", ".m4v", ".avi", ".webm"}
# HLG / PQ / wide-gamut markers that mean "tone-map me"
HDR_MARKERS = ("arib-std-b67", "smpte2084", "bt2020")
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def ffmpeg_exe() -> str:
    from shutil import which
    exe = which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("No ffmpeg. Run:  pip install imageio-ffmpeg")


def probe(ff: str, clip: Path):
    """Return (duration_seconds, is_hdr) by parsing ffmpeg -i stderr."""
    err = subprocess.run([ff, "-hide_banner", "-i", str(clip)],
                         capture_output=True, text=True).stderr
    dur = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", err)
    if m:
        h, mn, s = m.groups()
        dur = int(h) * 3600 + int(mn) * 60 + float(s)
    vline = next((l for l in err.splitlines() if "Video:" in l), "")
    is_hdr = any(mk in vline.lower() for mk in HDR_MARKERS)
    return dur, is_hdr


def process_clip(ff, clip, out_dir, n_frames, fps, img_size):
    dur, is_hdr = probe(ff, clip)
    if dur <= 0:
        print(f"[skip] {clip.name}: no duration")
        return None
    # Fast-seek extraction: decode only ~1 frame per sample point (-ss before -i),
    # ~100x faster than a full-decode fps filter on long 4K HEVC.
    if fps:
        n = max(1, int(dur * fps))
    else:
        n = n_frames
    vf = (f"{TONEMAP},scale={img_size}:-1") if is_hdr else f"scale={img_size}:-1"
    stem = clip.stem.replace(" ", "_")
    written = 0
    for i in range(n):
        t = dur * (i + 0.5) / n                     # evenly spaced, mid-bin
        dst = out_dir / f"{stem}_{i + 1:03d}.jpg"
        subprocess.run([ff, "-y", "-nostdin", "-ss", f"{t:.3f}", "-i", str(clip),
                        "-frames:v", "1", "-vf", vf, "-qscale:v", "3",
                        "-loglevel", "error", str(dst)],
                       capture_output=True, text=True)
        if dst.exists():
            written += 1
    frames = sorted(out_dir.glob(f"{stem}_*.jpg"))
    print(f"[ok] {clip.name:24s} dur={dur:6.1f}s  hdr={is_hdr}  -> {written} frames")
    return {"name": stem, "source": str(clip), "duration_s": round(dur, 1),
            "hdr": is_hdr, "prefix": f"{stem}_", "count": len(frames),
            "frames": [str(p) for p in frames]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("processed"))
    ap.add_argument("--frames-per-clip", type=int, default=48,
                    help="evenly-spaced frames per clip (equal n = fair per-clip compare)")
    ap.add_argument("--fps", type=float, default=0.0, help="override: fixed sampling fps")
    ap.add_argument("--img-size", type=int, default=768)
    args = ap.parse_args()

    ff = ffmpeg_exe()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in args.clips_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not clips:
        sys.exit(f"No clips in {args.clips_dir}")

    manifest = []
    for c in clips:
        r = process_clip(ff, c, args.out_dir, args.frames_per_clip, args.fps, args.img_size)
        if r:
            manifest.append(r)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] {sum(m['count'] for m in manifest)} frames -> {args.out_dir}/")
    print("\n--- GROUPS spec for the benchmark workflow ---")
    for m in manifest:
        print(f"  {{ name: '{m['name']}', prefix: '{m['prefix']}', "
              f"count: {m['count']}, hdr: {m['hdr']}, bucket: 'user' }},")


if __name__ == "__main__":
    main()
