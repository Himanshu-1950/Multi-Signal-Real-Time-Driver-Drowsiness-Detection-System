# ================================================================
#  MULTI-SIGNAL DROWSINESS DETECTION — SCRATCH REWRITE
#  Simple, reliable, auto-calibrates to YOUR face
#
#  INSTALL:
#    pip install mediapipe==0.10.9
#    pip install opencv-python pygame numpy tensorflow
#
#  RUN:
#    python detect_advanced.py
#
#  HOW IT WORKS:
#    - Looks at your face for 3 seconds to learn YOUR values
#    - Then detects: eyes closed, yawning, head tilt
#    - Alarm rings when drowsy, stops when alert
# ================================================================

import cv2
import numpy as np
import mediapipe as mp
import pygame
import os, sys, math, wave, struct, csv
from datetime import datetime

# ── Check MediaPipe ────────────────────────────────────────
try:
    _ = mp.solutions.face_mesh
except AttributeError:
    print("❌ Run: pip uninstall mediapipe -y && pip install mediapipe==0.10.9")
    sys.exit(1)

# ── Generate alarm sound ───────────────────────────────────
if not os.path.exists('alarm.wav'):
    sr = 44100
    s  = [int(32767 * math.sin(2*math.pi*1000*t/sr)) for t in range(sr)]
    with wave.open('alarm.wav','w') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(struct.pack('<'+'h'*len(s),*s))

pygame.mixer.init()
sound    = pygame.mixer.Sound('alarm.wav')
alarm_on = False

# ── Load TF only for yawn CNN ──────────────────────────────
model_yawn = None
if os.path.exists('cnnYawn.keras'):
    try:
        from tensorflow.keras.models import load_model
        model_yawn = load_model('cnnYawn.keras')
        print("✅ cnnYawn.keras loaded!")
    except:
        print("⚠️  Could not load cnnYawn.keras — using MAR only")
else:
    print("⚠️  cnnYawn.keras not found — using MAR only")

# ── MediaPipe ──────────────────────────────────────────────
mp_mesh   = mp.solutions.face_mesh
face_mesh = mp_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ── Key landmark indices ───────────────────────────────────
# Eye landmarks — top and bottom points only (simplest approach)
# Left eye:  top=386, bottom=374, left=362, right=263
# Right eye: top=159, bottom=145, left=33,  right=133
L_TOP, L_BOT, L_LEFT, L_RIGHT = 386, 374, 362, 263
R_TOP, R_BOT, R_LEFT, R_RIGHT = 159, 145,  33, 133

# Mouth landmarks — top, bottom, left, right
M_TOP, M_BOT, M_LEFT, M_RIGHT = 13, 14, 78, 308

# Head tilt — left eye corner, right eye corner
H_LEFT, H_RIGHT = 33, 263

# ── Webcam ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam!"); sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font = cv2.FONT_HERSHEY_SIMPLEX

# ── Alert thresholds (will be calibrated) ─────────────────
EYE_CLOSED_RATIO = 0.20   # eye height/width ratio — below = closed
YAWN_OPEN_RATIO  = 0.40   # mouth height/width ratio — above = yawning
TILT_DEG         = 15.0   # head tilt degrees

# ── Alert scores ───────────────────────────────────────────
WARN_SCORE  = 8
ALARM_SCORE = 15
CRIT_SCORE  = 25

eye_score  = 0
yawn_score = 0
tilt_score = 0

# ── CSV log ────────────────────────────────────────────────
LOG = 'drowsiness_log.csv'
if not os.path.exists(LOG):
    with open(LOG,'w',newline='') as f:
        csv.writer(f).writerow(
            ['Time','Level','Eyes','Yawn','Head','EyeR','MouthR','Tilt'])

def log_event(lvl, eyes, yawn, head, er, mr, tilt):
    with open(LOG,'a',newline='') as f:
        csv.writer(f).writerow([
            datetime.now().strftime('%H:%M:%S'),
            lvl,
            'closed' if eyes else 'open',
            'yes'    if yawn else 'no',
            'tilted' if head else 'ok',
            f'{er:.3f}', f'{mr:.3f}', f'{tilt:.1f}'
        ])

# ── Helper: get x,y from landmark ─────────────────────────
def lxy(lm, idx, W, H):
    return int(lm[idx].x*W), int(lm[idx].y*H)

# ── Helper: distance between 2 points ─────────────────────
def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ── Helper: eye open ratio = height / width ────────────────
def eye_ratio(lm, top, bot, left, right, W, H):
    t = lxy(lm, top,   W, H)
    b = lxy(lm, bot,   W, H)
    l = lxy(lm, left,  W, H)
    r = lxy(lm, right, W, H)
    height = dist(t, b)
    width  = dist(l, r) + 1e-6
    return height / width

# ── Helper: mouth open ratio = height / width ─────────────
def mouth_ratio(lm, W, H):
    t = lxy(lm, M_TOP,   W, H)
    b = lxy(lm, M_BOT,   W, H)
    l = lxy(lm, M_LEFT,  W, H)
    r = lxy(lm, M_RIGHT, W, H)
    height = dist(t, b)
    width  = dist(l, r) + 1e-6
    return height / width

# ── Helper: head tilt angle ────────────────────────────────
def head_tilt(lm, W, H):
    l = lxy(lm, H_LEFT,  W, H)
    r = lxy(lm, H_RIGHT, W, H)
    dx = r[0] - l[0]
    dy = r[1] - l[1]
    return abs(math.degrees(math.atan2(dy, dx+1e-6)))

# ── Draw bar helper ────────────────────────────────────────
def draw_bar(frame, x, y, w, h, val, maxv, col, label):
    fill = min(int(w*val/maxv), w)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(40,40,40),-1)
    if fill>0:
        cv2.rectangle(frame,(x,y),(x+fill,y+h),col,-1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(150,150,150),1)
    cv2.putText(frame,f'{label}:{val}/{maxv}',
                (x,y-5),font,0.38,(180,180,180),1)

# ══════════════════════════════════════════════════════════
#  STEP 1 — AUTO CALIBRATION
#  Sit normally, eyes open, mouth closed for 3 seconds
# ══════════════════════════════════════════════════════════
print("\n⏳ AUTO CALIBRATION STARTING...")
print("   Sit normally — eyes OPEN, mouth CLOSED")
print("   Hold still for 3 seconds...\n")

eye_samples   = []
mouth_samples = []
calib_done    = False
calib_count   = 0
CALIB_FRAMES  = 60   # 2 seconds

while not calib_done:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        er = (eye_ratio(lm, L_TOP, L_BOT, L_LEFT, L_RIGHT, W, H) +
              eye_ratio(lm, R_TOP, R_BOT, R_LEFT, R_RIGHT, W, H)) / 2.0
        mr = mouth_ratio(lm, W, H)

        eye_samples.append(er)
        mouth_samples.append(mr)
        calib_count += 1

        # Progress bar
        prog = int(W * calib_count / CALIB_FRAMES)
        cv2.rectangle(frame,(0,H-30),(prog,H),(0,255,100),-1)
        cv2.rectangle(frame,(0,H-30),(W,H),(100,100,100),2)

        secs = (CALIB_FRAMES - calib_count) // 30 + 1
        cv2.putText(frame, f"Calibrating... {secs}s",
                    (W//2-130,H//2-20),font,1.0,(0,255,255),2)
        cv2.putText(frame, "Eyes OPEN  Mouth CLOSED",
                    (W//2-160,H//2+20),font,0.65,(0,255,255),1)
        cv2.putText(frame, f"EyeR:{er:.3f}  MouthR:{mr:.3f}",
                    (10,30),font,0.55,(200,200,200),1)

        if calib_count >= CALIB_FRAMES:
            calib_done = True
    else:
        cv2.putText(frame, "No face! Move closer to camera",
                    (W//2-200,H//2),font,0.8,(0,0,255),2)

    cv2.imshow("Multi-Signal Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release(); cv2.destroyAllWindows(); sys.exit(0)

# Set thresholds from calibration
if len(eye_samples) > 10:
    avg_eye   = np.mean(eye_samples)
    avg_mouth = np.mean(mouth_samples)

    # Eye closed = ratio drops to ~50% of open
    EYE_CLOSED_RATIO = avg_eye * 0.60

    # Yawn = mouth opens to ~2.5x normal closed position
    YAWN_OPEN_RATIO  = avg_mouth * 2.5

    print(f"✅ Calibration complete!")
    print(f"   Normal eye ratio   : {avg_eye:.3f}")
    print(f"   Normal mouth ratio : {avg_mouth:.3f}")
    print(f"   Eye CLOSED below   : {EYE_CLOSED_RATIO:.3f}")
    print(f"   Yawn OPEN above    : {YAWN_OPEN_RATIO:.3f}")
else:
    print("⚠️  Calibration failed! Using defaults.")

# ══════════════════════════════════════════════════════════
#  STEP 2 — MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════
print("\n🚀 Detection running! Press Q to quit.")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = face_mesh.process(rgb)

    frame_count   += 1
    eyes_closed    = False
    yawning        = False
    head_tilted    = False
    eye_r_val      = 0.0
    mouth_r_val    = 0.0
    tilt_val       = 0.0

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        # ── Eye ratio ──────────────────────────────────────
        left_er  = eye_ratio(lm, L_TOP, L_BOT, L_LEFT, L_RIGHT, W, H)
        right_er = eye_ratio(lm, R_TOP, R_BOT, R_LEFT, R_RIGHT, W, H)
        eye_r_val = (left_er + right_er) / 2.0
        eyes_closed = eye_r_val < EYE_CLOSED_RATIO

        # Draw eye points
        for idx in [L_TOP,L_BOT,L_LEFT,L_RIGHT,
                    R_TOP,R_BOT,R_LEFT,R_RIGHT]:
            px,py = lxy(lm,idx,W,H)
            col   = (0,0,255) if eyes_closed else (0,255,150)
            cv2.circle(frame,(px,py),2,col,-1)

        # ── Mouth ratio ────────────────────────────────────
        mouth_r_val = mouth_ratio(lm, W, H)
        yawning     = mouth_r_val > YAWN_OPEN_RATIO

        # Draw mouth points
        for idx in [M_TOP, M_BOT, M_LEFT, M_RIGHT]:
            px,py = lxy(lm,idx,W,H)
            col   = (0,165,255) if yawning else (255,200,0)
            cv2.circle(frame,(px,py),3,col,-1)

        # Draw mouth line
        t = lxy(lm,M_TOP,W,H);  b = lxy(lm,M_BOT,W,H)
        l = lxy(lm,M_LEFT,W,H); r = lxy(lm,M_RIGHT,W,H)
        mc = (0,165,255) if yawning else (200,200,0)
        cv2.line(frame,l,r,mc,1)
        cv2.line(frame,t,b,mc,1)

        # ── Head tilt ──────────────────────────────────────
        tilt_val    = head_tilt(lm, W, H)
        head_tilted = tilt_val > TILT_DEG

        lp = lxy(lm,H_LEFT,W,H)
        rp = lxy(lm,H_RIGHT,W,H)
        tc = (0,0,255) if head_tilted else (0,200,255)
        cv2.line(frame, lp, rp, tc, 2)

    else:
        cv2.putText(frame,"No face detected",
                    (10,35),font,0.8,(0,165,255),2)

    # ── Update scores ──────────────────────────────────────
    # Each signal scores independently so yawn/tilt can trigger alone
    eye_score  = eye_score+1  if eyes_closed  else 0
    yawn_score = yawn_score+1 if yawning      else max(yawn_score-1,0)
    tilt_score = tilt_score+1 if head_tilted  else max(tilt_score-1,0)

    total = eye_score + (yawn_score//2) + (tilt_score//2)

    # ── Alert level ────────────────────────────────────────
    reasons = []
    if eye_score  >= WARN_SCORE: reasons.append("Eyes")
    if yawn_score >= WARN_SCORE: reasons.append("Yawn")
    if tilt_score >= WARN_SCORE: reasons.append("Head")
    reason_str = "+".join(reasons) if reasons else "None"

    if   total >= CRIT_SCORE:  lvl,msg,col = 3,"CRITICAL! PULL OVER!", (0,0,255)
    elif total >= ALARM_SCORE: lvl,msg,col = 2,"DROWSINESS ALERT!",    (0,50,255)
    elif total >= WARN_SCORE:  lvl,msg,col = 1,"WARNING: Stay Alert",  (0,165,255)
    else:                      lvl,msg,col = 0,"",                     (0,255,0)

    # ── Alarm control ──────────────────────────────────────
    # FIX: if alarm is on and eyes just opened, stop immediately
    # but only if yawn and tilt are also below warning threshold
    # This keeps yawn/tilt independent but lets eyes stop the alarm fast
    if alarm_on and not eyes_closed and eye_score == 0:
        if yawn_score < WARN_SCORE and tilt_score < WARN_SCORE:
            sound.stop(); pygame.mixer.stop(); alarm_on = False

    if lvl >= 2:
        if not alarm_on:
            sound.play(-1); alarm_on = True
        if frame_count % 30 == 0:
            log_event(lvl,eyes_closed,yawning,head_tilted,
                      eye_r_val,mouth_r_val,tilt_val)
    else:
        if alarm_on:
            sound.stop(); pygame.mixer.stop(); alarm_on = False

    # ── Alert display ──────────────────────────────────────
    if lvl == 3:
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(W,H),(0,0,180),-1)
        cv2.addWeighted(ov,0.3,frame,0.7,0,frame)
        cv2.putText(frame,msg,(W//2-200,H//2),font,1.1,(255,255,255),3)
    elif lvl == 2:
        cv2.rectangle(frame,(0,0),(W,H),(0,0,255),5)
        cv2.putText(frame,msg,(W//2-175,50),font,1.0,(0,0,255),3)
    elif lvl == 1:
        cv2.rectangle(frame,(0,0),(W,H),(0,165,255),3)
        cv2.putText(frame,msg,(W//2-160,40),font,0.8,(0,165,255),2)

    # ── Status display ─────────────────────────────────────
    ec = (0,0,255)   if eyes_closed  else (0,255,0)
    yc = (0,165,255) if yawning      else (0,255,0)
    tc = (0,100,255) if head_tilted  else (0,255,0)

    cv2.putText(frame,
        f"Eyes : {'CLOSED' if eyes_closed else 'OPEN  '}"
        f"  ratio:{eye_r_val:.3f}  thresh:{EYE_CLOSED_RATIO:.3f}",
        (10,H-105),font,0.50,ec,2)
    cv2.putText(frame,
        f"Yawn : {'YES   ' if yawning else 'NO    '}"
        f"  ratio:{mouth_r_val:.3f}  thresh:{YAWN_OPEN_RATIO:.3f}",
        (10,H-72), font,0.50,yc,2)
    cv2.putText(frame,
        f"Head : {'TILT  ' if head_tilted else 'OK    '}"
        f"  {tilt_val:.1f}deg  thresh:{TILT_DEG:.0f}deg",
        (10,H-40), font,0.50,tc,2)

    # Score bar
    bx = W-215
    bc = (0,255,0) if total<WARN_SCORE else \
         (0,165,255) if total<ALARM_SCORE else (0,0,255)
    draw_bar(frame,bx,H-35,200,14,
             min(total,CRIT_SCORE),CRIT_SCORE,bc,"Score")

    cv2.putText(frame,"Press Q to quit",
                (W-150,18),font,0.38,(130,130,130),1)

    cv2.imshow("Multi-Signal Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ────────────────────────────────────────────────
sound.stop()
pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print(f"\n✅ Stopped. Log: {LOG}")
