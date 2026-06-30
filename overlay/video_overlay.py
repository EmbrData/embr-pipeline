#!/usr/bin/env python3
"""
video_overlay.py - per-frame 3D hand-pose overlay across a clip segment (VIDEO mode).
Reads seg/*.jpg (ordered), writes seg_out/*.png. Encode to mp4 separately with ffmpeg.
Hand pose = real MediaPipe Hand Landmarker; caption = VLM (Qwen3-VL) stand-in.
"""
import cv2, glob, os, mediapipe as mp
from mediapipe.tasks import python as mptask
from mediapipe.tasks.python import vision as mpv
from PIL import Image, ImageDraw, ImageFont

SRC = "seg"; DST = "seg_out"; MODEL = "models/hand_landmarker.task"; FPS = 30
ACTION = "Fastening interior assist-grip with powered driver"
META = "3D hand pose: MediaPipe (21 kp/hand)   |   Action + objects: VLM (Qwen3-VL)   |   robot-ready"
OBJECTS = ["assist-grip handle", "mounting bracket", "driver bit"]
CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
HCOL = {"Left": (57, 255, 20), "Right": (0, 200, 255)}   # stable color per actual hand

def _font(sz):
    for p in ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

files = sorted(glob.glob(f"{SRC}/*.jpg"))
if not files: raise SystemExit("no frames in seg/")
os.makedirs(DST, exist_ok=True)
probe = cv2.imread(files[0]); h, w = probe.shape[:2]
F_CHIP, F_BADGE, F_ACT, F_META, F_OBJ = _font(max(20, w//40)), _font(max(22, w//34)), \
    _font(max(28, w//24)), _font(max(16, w//50)), _font(max(16, w//52))

opts = mpv.HandLandmarkerOptions(
    base_options=mptask.BaseOptions(model_asset_path=MODEL),
    running_mode=mpv.RunningMode.VIDEO, num_hands=2,
    min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)
det = mpv.HandLandmarker.create_from_options(opts)

def fit(text, maxw, start, draw, minsz=14):
    s = start
    while s > minsz:
        f = _font(s)
        if draw.textlength(text, font=f) <= maxw: return f
        s -= 2
    return _font(minsz)

for i, f in enumerate(files):
    rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i*1000/FPS))
    pil = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(pil, "RGBA")
    for hi, lms in enumerate(res.hand_landmarks):
        side = ""
        try: side = res.handedness[hi][0].category_name
        except Exception: pass
        col = HCOL.get(side, (57, 255, 20))
        pts = [(int(l.x*w), int(l.y*h)) for l in lms]
        for a, b in CONN:
            draw.line([pts[a], pts[b]], fill=col+(235,), width=max(2, w//360))
        for (x, y) in pts:
            r = max(3, w//300)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 255, 255, 255), outline=col+(255,))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, y0 = min(xs)-14, min(ys)-14
        draw.rectangle([x0, y0, max(xs)+14, max(ys)+14], outline=col+(255,), width=max(2, w//560))
        txt = f"{side or 'hand'} · 21 kp"
        tb = draw.textbbox((0, 0), txt, font=F_CHIP); tw, th = tb[2]-tb[0], tb[3]-tb[1]
        cy = max(th+12, y0)
        draw.rectangle([max(0, x0), cy-th-10, max(0, x0)+tw+12, cy], fill=(0, 0, 0, 200))
        draw.text((max(0, x0)+8, cy-th-7), txt, font=F_CHIP, fill=(255, 255, 255, 255))
    # badge
    tb = draw.textbbox((0, 0), "EMBR auto-labeling", font=F_BADGE)
    draw.rectangle([0, 0, tb[2]-tb[0]+40, tb[3]-tb[1]+30], fill=(20, 30, 50, 225))
    draw.text((20, 12), "EMBR auto-labeling", font=F_BADGE, fill=(120, 230, 160, 255))
    # caption bar
    barh = int(h*0.17); pad = int(w*0.03)
    draw.rectangle([0, h-barh, w, h], fill=(8, 12, 22, 228))
    draw.rectangle([0, h-barh, w, h-barh+4], fill=(57, 255, 20, 255))
    draw.text((pad, h-barh+int(barh*0.13)), ACTION, font=fit(ACTION, w-2*pad, max(28, w//24), draw), fill=(255,)*3+(255,))
    draw.text((pad, h-barh+int(barh*0.49)), META, font=F_META, fill=(150, 235, 180, 255))
    draw.text((pad, h-barh+int(barh*0.72)), "objects:  " + "   ·   ".join(OBJECTS), font=F_OBJ, fill=(170, 200, 240, 255))
    pil.save(f"{DST}/{i:04d}.png")

print(f"rendered {len(files)} frames -> {DST}/ ({w}x{h})")
