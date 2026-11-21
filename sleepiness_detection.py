"""Real-time drowsiness detection from a webcam using the eye aspect ratio (EAR).

dlib finds 68 facial landmarks per frame; the six landmarks around each eye give
the EAR (Soukupova & Cech, 2016). When the average EAR stays below a threshold
for several consecutive frames, the eyes are treated as closed and a spoken
alert is played.

    python sleepiness_detection.py --landmarks shape_predictor_68_face_landmarks.dat
"""
from __future__ import annotations

import argparse
import time

import cv2
import dlib
import pyttsx3
from scipy.spatial import distance

RIGHT_EYE = range(36, 42)  # landmark indices in the 68-point dlib model
LEFT_EYE = range(42, 48)


def eye_aspect_ratio(eye: list[tuple[int, int]]) -> float:
    """Ratio of eye height to width; drops towards 0 as the eye closes."""
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c)


def eye_points(landmarks, indices) -> list[tuple[int, int]]:
    return [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]


def draw_contour(frame, points, color) -> None:
    for i, p in enumerate(points):
        cv2.line(frame, p, points[(i + 1) % len(points)], color, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--landmarks", default="shape_predictor_68_face_landmarks.dat",
                    help="path to dlib's 68-point landmark model")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--threshold", type=float, default=0.15, help="EAR below this counts as closed")
    ap.add_argument("--frames", type=int, default=15,
                    help="consecutive closed frames before alerting (filters out normal blinks)")
    ap.add_argument("--cooldown", type=float, default=3.0, help="seconds between spoken alerts")
    args = ap.parse_args()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.landmarks)
    voice = pyttsx3.init()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}; try --camera 1")

    closed_frames = 0
    last_alert = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for face in detector(gray):
                landmarks = predictor(gray, face)
                left, right = eye_points(landmarks, LEFT_EYE), eye_points(landmarks, RIGHT_EYE)
                draw_contour(frame, left, (0, 255, 0))
                draw_contour(frame, right, (255, 255, 0))

                ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2
                closed_frames = closed_frames + 1 if ear < args.threshold else 0
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if closed_frames >= args.frames:
                    cv2.putText(frame, "DROWSINESS ALERT", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if time.time() - last_alert > args.cooldown:
                        voice.say("Wake up")
                        voice.runAndWait()
                        last_alert = time.time()

            cv2.imshow("Drowsiness detection (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
