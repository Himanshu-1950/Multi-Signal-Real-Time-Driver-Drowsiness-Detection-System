# Multi-Signal Drowsiness Detection System 🚗😴

## Overview
Advanced real-time driver drowsiness detection using **3 signals**:
- **Eye Closure** (MediaPipe landmarks)
- **Yawning** (MediaPipe + optional CNN model)
- **Head Tilt** (MediaPipe landmarks)

**Auto-calibrates** to YOUR face in 3 seconds. Sounds alarm on detection. Logs events.

## Features
- ✅ **No training needed** - learns your baseline automatically
- ✅ **Multi-signal scoring** (eyes + yawn + tilt)
- ✅ **Progressive alerts** (warning → alarm → CRITICAL)
- ✅ **Webcam ready** - works with any camera
- ✅ **CSV logging** for analysis
- ✅ **Self-contained** - generates alarm sound if missing

## Quick Start
```bash
pip install -r requirements.txt
python detect_advanced.py
```
- Calibration: Sit normally (eyes OPEN, mouth CLOSED) for 3s
- Press **Q** to quit

## Installation
```bash
# Recommended: virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Or global (already done)
pip install mediapipe==0.10.9 opencv-python pygame numpy tensorflow
```

**Note:** mediapipe **exactly 0.10.9** - FaceMesh compatibility.

## Usage
1. Run `python detect_advanced.py`
2. **Calibration** window: Hold position 3s
3. **Detection** window shows:
   - Live metrics (EyeR, MouthR, Tilt)
   - Score bar (green→yellow→red)
   - Alerts/Alarms
4. Logs → `drowsiness_log.csv`

## How It Works
```
1. CALIBRATE: 60 frames → your normal eye/mouth ratios
2. DETECT:
   │ Eyes Closed (ratio <60% normal) → +1 score/sec
   │ Yawn (mouth >2.5x normal)      → +1 score/sec  
   │ Head Tilt (>15°)               → +1 score/sec
3. ALERTS:
   8+ pts: Warning    (yellow border)
   15+ pts: Alarm     (red border + beep)
   25+ pts: CRITICAL! (red overlay)
```

## Files
| File | Purpose |
|------|---------|
| `detect_advanced.py` | Main detection app |
| `test_model.py` | Test CNN eye model |
| `cnnYawn.keras` | Yawn classification (optional) |
| `cnnEye.keras` | Eye classification (test only) |
| `alarm.wav` | Alert sound (auto-generated) |
| `requirements.txt` | Dependencies |
| `drowsiness_log.csv` | Event logs |
| `view_log.py` | Log viewer |
| `TODO.md` | Run progress |

## Thresholds (Auto-Set)
```
Eye Closed:  < 60% of your normal eye aspect ratio
Yawn Open:   > 2.5x your normal mouth ratio  
Head Tilt:   > 15° from eye line
```

## Troubleshooting
```
❌ "Cannot open webcam": Check camera privacy settings
❌ "mediapipe AttributeError": pip uninstall mediapipe -y && pip install mediapipe==0.10.9
❌ No sound: pygame speakers muted?
✅ Script self-checks mediapipe version
```

## Logs Example
```
Time,Level,Eyes,Yawn,Head,EyeR,MouthR,Tilt
21:01:36,2,closed,no,ok,0.120,0.001,5.2
```

## Tech Stack
- **MediaPipe FaceMesh** (landmarks)
- **OpenCV** (video)
- **Pygame** (alarm)
- **TensorFlow/Keras** (CNN yawn)
- **Python 3.11+**

**Stop drowsiness before it stops you! 🚀**

