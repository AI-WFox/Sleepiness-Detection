# Drowsiness Detection

Real-time drowsiness detection from a webcam. The program tracks both eyes with
dlib's 68 facial landmarks, computes the **eye aspect ratio (EAR)** every frame,
and plays a spoken alert when the eyes stay closed for longer than a normal blink.

Typical use: driver or night-shift fatigue warning on a laptop or a small board
with a camera.

## How it works

1. dlib's HOG face detector finds faces in each grayscale frame.
2. The 68-point shape predictor locates the six landmarks around each eye.
3. EAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 ‖p1 − p4‖), averaged over both eyes
   (Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks", CVWW 2016).
   Open eyes sit around 0.25 to 0.35; closed eyes drop below about 0.15.
4. If EAR stays under the threshold for `--frames` consecutive frames
   (about half a second at 30 fps), the frame shows an alert and a voice says
   "Wake up". A cooldown stops the alert from repeating every frame.

## Setup

```bash
pip install -r requirements.txt
```

Download the landmark model and unpack it next to the script:

```bash
curl -LO http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

`dlib` needs CMake and a C++ compiler to build; on Windows, a prebuilt wheel
for your Python version is the easiest route.

## Run

```bash
python sleepiness_detection.py
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--landmarks` | `shape_predictor_68_face_landmarks.dat` | path to the dlib model |
| `--camera` | `0` | webcam index (try `1` for an external camera) |
| `--threshold` | `0.15` | EAR below this counts as closed |
| `--frames` | `15` | consecutive closed frames before alerting |
| `--cooldown` | `3.0` | seconds between spoken alerts |

Press `q` to quit.

## Limitations

- A fixed EAR threshold does not suit every face; calibrating per user
  (for example 70% of that person's open-eye EAR) is more robust.
- The HOG detector struggles in low light and with strong head turns; glasses
  with reflections can hide the eye landmarks.
