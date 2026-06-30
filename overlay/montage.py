#!/usr/bin/env python3
"""
montage.py - 15s side-by-side montage: RAW | EMBR AUTO-LABELED.
Clean settings (same as the 5s clip): MediaPipe Hand Landmarker, VIDEO tracking mode,
confidence 0.3, handedness coloring. No mirror-merge / skin-filter / temporal-fill.
Hand pose is real MediaPipe output; caption = VLM (Qwen3-VL) stand-in.
"""
import cv2, glob, os, mediapipe as mp
from mediapipe.tasks import python as mptask
from mediapipe.tasks.python import vision as mpv
from PIL import Image, ImageDraw, ImageFont

MODEL = "models/hand_landmarker.task"; FPS = 30; OUT = "montage"; CONF = 0.3
WINDOWS = [
    {"dir": "seg",  "action": "Fastening interior assist-grip with powered driver",
     "objects": ["assist-grip handle", "mounting bracket", "driver bit"]},
    {"dir": "seg7", "action": "Driving fastener with pneumatic air tool",
     "objects": ["body panel", "pneumatic air tool"]},
    {"dir": "seg5", "action": "Fitting retainer clips into airbag trim panel",
     "objects": ["airbag trim panel", "retainer clips"]},
]
CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
HCOL = {"Left": (57, 255, 20), "Right": (0, 200, 255)}
META = "3D hand pose: MediaPipe (21 kp/hand)   |   Action + objects: VLM (Qwen3-VL)   |   robot-ready"

def _f(sz):
    for p in ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

pw, ph = Image.open(sorted(glob.glob(f"{WINDOWS[0]['dir']}/*.jpg"))[0]).size
HEAD = 72; W, H = pw*2, ph+HEAD
F_CHIP, F_BADGE, F_META, F_OBJ, F_HEAD = _f(max(18,pw//40)), _f(max(20,pw//34)), _f(max(15,pw//50)), _f(max(15,pw//52)), _f(30)

def fit(text, maxw, start, draw, mn=14):
    s = start
    while s > mn:
        f = _f(s)
        if draw.textlength(text, font=f) <= maxw: return f
        s -= 2
    return _f(mn)

def draw_overlay(rgb, res, action, objects):
    pil = Image.fromarray(rgb).convert("RGB"); draw = ImageDraw.Draw(pil, "RGBA"); w, h = pw, ph
    for hi, lms in enumerate(res.hand_landmarks):
        side = ""
        try: side = res.handedness[hi][0].category_name
        except Exception: pass
        col = HCOL.get(side, (57, 255, 20))
        pts = [(int(l.x*w), int(l.y*h)) for l in lms]
        for a, b in CONN: draw.line([pts[a], pts[b]], fill=col+(235,), width=max(2, w//360))
        for (x, y) in pts:
            r = max(3, w//300); draw.ellipse([x-r, y-r, x+r, y+r], fill=(255,255,255,255), outline=col+(255,))
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; x0,y0=min(xs)-14,min(ys)-14
        draw.rectangle([x0,y0,max(xs)+14,max(ys)+14], outline=col+(255,), width=max(2,w//560))
        t=f"{side or 'hand'} · 21 kp"; tb=draw.textbbox((0,0),t,font=F_CHIP); tw,th=tb[2]-tb[0],tb[3]-tb[1]
        cyy=max(th+12,y0); draw.rectangle([max(0,x0),cyy-th-10,max(0,x0)+tw+12,cyy], fill=(0,0,0,200))
        draw.rectangle([max(0,x0),cyy-th-10,max(0,x0)+4,cyy], fill=col+(255,))
        draw.text((max(0,x0)+8,cyy-th-7), t, font=F_CHIP, fill=(255,255,255,255))
    tb=draw.textbbox((0,0),"EMBR auto-labeling",font=F_BADGE)
    draw.rectangle([0,0,tb[2]-tb[0]+36,tb[3]-tb[1]+26], fill=(20,30,50,225))
    draw.text((18,10),"EMBR auto-labeling",font=F_BADGE,fill=(120,230,160,255))
    barh=int(h*0.17); pad=int(w*0.03)
    draw.rectangle([0,h-barh,w,h], fill=(8,12,22,230)); draw.rectangle([0,h-barh,w,h-barh+4], fill=(57,255,20,255))
    draw.text((pad,h-barh+int(barh*0.13)), action, font=fit(action,w-2*pad,max(26,w//24),draw), fill=(255,255,255,255))
    draw.text((pad,h-barh+int(barh*0.49)), META, font=F_META, fill=(150,235,180,255))
    draw.text((pad,h-barh+int(barh*0.72)), "objects:  "+"   ·   ".join(objects), font=F_OBJ, fill=(170,200,240,255))
    return pil

os.makedirs(OUT, exist_ok=True)
gidx = 0
for win in WINDOWS:
    files = sorted(glob.glob(f"{win['dir']}/*.jpg")); n = len(files)
    det = mpv.HandLandmarker.create_from_options(mpv.HandLandmarkerOptions(
        base_options=mptask.BaseOptions(model_asset_path=MODEL), running_mode=mpv.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=CONF, min_hand_presence_confidence=CONF, min_tracking_confidence=CONF))
    two = 0
    for i, f in enumerate(files):
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i*1000/FPS))
        two += (len(res.hand_landmarks) >= 2)
        raw = Image.fromarray(rgb).convert("RGB")
        over = draw_overlay(rgb, res, win["action"], win["objects"])
        canvas = Image.new("RGB", (W, H), (8, 12, 22)); d = ImageDraw.Draw(canvas)
        canvas.paste(raw, (0, HEAD)); canvas.paste(over, (pw, HEAD))
        d.line([(pw, HEAD), (pw, H)], fill=(57, 255, 20), width=3)
        d.rectangle([0, 0, W, HEAD], fill=(15, 22, 38))
        for label, lx in [("RAW CAPTURE", pw//2), ("EMBR AUTO-LABELED", pw + pw//2)]:
            tb = d.textbbox((0, 0), label, font=F_HEAD)
            d.text((lx-(tb[2]-tb[0])//2, (HEAD-(tb[3]-tb[1]))//2 - 4), label, font=F_HEAD,
                   fill=(255, 255, 255) if "RAW" in label else (120, 230, 160))
        canvas.save(f"{OUT}/{gidx:05d}.png"); gidx += 1
    print(f"  {win['dir']}: {n} frames | two-hand {100*two//n}%")
print(f"total montage frames: {gidx}  ({W}x{H})")
