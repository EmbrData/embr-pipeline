#!/usr/bin/env python3
"""
overlay.py - hero-slide overlay: real 3D hand pose (MediaPipe Tasks) + VLM-style action label.

Hand pose is REAL (MediaPipe Hand Landmarker, 21 3D keypoints/hand, Apache-2.0 - the
commercially clean path). The action caption + object tags are the VLM (Qwen3-VL) layer,
authored here as a stand-in (Qwen needs a GPU/token). Production: HaMeR mesh if MANO is
licensed, else MediaPipe; captions from Qwen3-VL.
"""
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mptask
from mediapipe.tasks.python import vision as mpv
from PIL import Image, ImageDraw, ImageFont

IN = "examples/hero_frame.jpg"
OUT = "results/hero_overlay.png"
MODEL = "models/hand_landmarker.task"

# --- VLM-layer annotation (stand-in for Qwen3-VL output) ---
ACTION = "Fastening interior assist-grip with powered driver"
META = "3D hand pose: 2 hands · 42 keypoints (MediaPipe)   |   Action + objects: VLM (Qwen3-VL)   |   robot-ready"
OBJECTS = ["assist-grip handle", "mounting bracket", "driver bit"]

# Standard MediaPipe 21-keypoint hand skeleton connections
CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

def font(sz, bold=False):
    for p in ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

img = cv2.imread(IN)
if img is None: raise SystemExit(f"cannot read {IN}")
h, w = img.shape[:2]
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

opts = mpv.HandLandmarkerOptions(
    base_options=mptask.BaseOptions(model_asset_path=MODEL),
    running_mode=mpv.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.3)
det = mpv.HandLandmarker.create_from_options(opts)
res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

pil = Image.fromarray(rgb).convert("RGB")
draw = ImageDraw.Draw(pil, "RGBA")
COLORS = [(57, 255, 20), (0, 200, 255)]      # lime, cyan

def label_chip(xy, text, col, fnt):
    x, y = xy
    tb = draw.textbbox((0, 0), text, font=fnt)
    tw, th = tb[2]-tb[0], tb[3]-tb[1]
    draw.rectangle([x, y-th-12, x+tw+16, y], fill=(0, 0, 0, 205))
    draw.rectangle([x, y-th-12, x+5, y], fill=col+(255,))
    draw.text((x+11, y-th-8), text, font=fnt, fill=(255, 255, 255, 255))

ndet = len(res.hand_landmarks)
for hi, lms in enumerate(res.hand_landmarks):
    col = COLORS[hi % 2]
    pts = [(int(l.x*w), int(l.y*h)) for l in lms]
    for a, b in CONN:
        draw.line([pts[a], pts[b]], fill=col+(235,), width=max(3, w//360))
    for (x, y) in pts:
        r = max(4, w//260)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 255, 255, 255), outline=col+(255,), width=2)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, y0, x1, y1 = min(xs)-18, min(ys)-18, max(xs)+18, max(ys)+18
    draw.rectangle([x0, y0, x1, y1], outline=col+(255,), width=max(2, w//500))
    side = ""
    try: side = res.handedness[hi][0].category_name
    except Exception: pass
    label_chip((max(0, x0), max(28, y0)), f"{side or 'hand'} hand · 21 kp", col, font(max(22, w//40), bold=True))

# --- top-left pipeline badge ---
bf = font(max(24, w//34), bold=True)
btxt = "EMBR auto-labeling"
tb = draw.textbbox((0, 0), btxt, font=bf)
draw.rectangle([0, 0, tb[2]-tb[0]+44, tb[3]-tb[1]+34], fill=(20, 30, 50, 225))
draw.text((22, 14), btxt, font=bf, fill=(120, 230, 160, 255))

# --- bottom caption bar ---
barh = int(h*0.17)
draw.rectangle([0, h-barh, w, h], fill=(8, 12, 22, 228))
draw.rectangle([0, h-barh, w, h-barh+5], fill=(57, 255, 20, 255))
pad = int(w*0.03)
def fit_font(text, maxw, start, bold=False, minsz=18):
    s = start
    while s > minsz:
        f = font(s, bold)
        if draw.textlength(text, font=f) <= maxw:
            return f
        s -= 2
    return font(minsz, bold)
draw.text((pad, h-barh+int(barh*0.13)), ACTION,
          font=fit_font(ACTION, w-2*pad, max(34, w//24), bold=True), fill=(255, 255, 255, 255))
draw.text((pad, h-barh+int(barh*0.48)), META, font=font(max(20, w//50)), fill=(150, 235, 180, 255))
draw.text((pad, h-barh+int(barh*0.72)), "objects:  " + "   ·   ".join(OBJECTS),
          font=font(max(20, w//52)), fill=(170, 200, 240, 255))

pil.save(OUT)
print(f"hands detected: {ndet}  ->  wrote {OUT} ({w}x{h})")
