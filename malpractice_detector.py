import cv2
import mediapipe as mp
import time
import csv
import os
import numpy as np
import math
from collections import deque

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=5, refine_landmarks=True)

cap = cv2.VideoCapture(0)

os.makedirs("evidence", exist_ok=True)

if not os.path.exists("log.csv"):
    with open("log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "event", "score"])

score = 0
MAX_SCORE = 20000

alert_text = ""
alert_time = 0

movement_history = deque(maxlen=10)

whitelist_set = False
reference_face = None
WHITELIST_THRESHOLD = 0.15
SKIP_SCORING = False


def log(event, score):
    with open("log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.ctime(), event, score])


def save(frame, label):
    path = "evidence/" + label + "_" + str(int(time.time())) + ".jpg"
    cv2.imwrite(path, frame)


def draw_box(frame, text, color, y):
    cv2.putText(frame, text, (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2)


def head_direction(landmarks, w, h):
    nose = landmarks[1]
    left = landmarks[33]
    right = landmarks[263]

    nx = nose.x * w
    lx = left.x * w
    rx = right.x * w

    if nx < lx:
        return "LEFT"
    if nx > rx:
        return "RIGHT"
    return "CENTER"


def eye_aspect(landmarks, w, h):
    left_top = landmarks[159].y * h
    left_bottom = landmarks[145].y * h
    right_top = landmarks[386].y * h
    right_bottom = landmarks[374].y * h

    left_ratio = abs(left_top - left_bottom)
    right_ratio = abs(right_top - right_bottom)

    return (left_ratio + right_ratio) / 2


def face_distance(lm1, lm2, w, h):
    total = 0
    count = 0

    for i in [1, 33, 263, 159, 145, 386, 374]:
        x1 = lm1[i].x * w
        y1 = lm1[i].y * h

        x2 = lm2[i].x * w
        y2 = lm2[i].y * h

        total += math.dist([x1, y1], [x2, y2])
        count += 1

    return total / count


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    lower_zone = frame[int(h * 0.25):h, 0:w]
    gray_lower = cv2.cvtColor(lower_zone, cv2.COLOR_BGR2GRAY)

    motion = cv2.mean(gray_lower)[0]
    movement_history.append(motion)

    motion_delta = 0
    if len(movement_history) > 2:
        motion_delta = abs(movement_history[-1] - movement_history[-2])

    if result.multi_face_landmarks:
        faces = result.multi_face_landmarks
        current_face = faces[0].landmark

        if not whitelist_set:
            reference_face = current_face
            whitelist_set = True

        if reference_face is not None:
            diff = face_distance(current_face, reference_face, w, h)

            if diff < WHITELIST_THRESHOLD:
                SKIP_SCORING = True
            else:
                SKIP_SCORING = False
        else:
            SKIP_SCORING = False

        if not SKIP_SCORING:

            if len(faces) > 1:
                score += 4
                alert_text = "MULTI FACE DETECTED"
                alert_time = time.time()
                log("multi_face", score)

            for face in faces:
                lm = face.landmark

                direction = head_direction(lm, w, h)

                if direction != "CENTER":
                    score += 1
                    alert_text = "HEAD MOVEMENT"
                    alert_time = time.time()
                    log("head_move", score)

                eye = eye_aspect(lm, w, h)

                if eye < 2:
                    score += 2
                    alert_text = "EYE CLOSED OR LOOKING DOWN"
                    alert_time = time.time()
                    log("eye_event", score)

    else:
        score += 5
        alert_text = "NO FACE DETECTED"
        alert_time = time.time()
        log("no_face", score)

    if motion_delta > 8:
        score += 2
        alert_text = "SUSPICIOUS DEVICE MOVEMENT"
        alert_time = time.time()
        log("motion_device", score)

    if np.mean(gray_lower) > 110:
        score += 2
        alert_text = "POSSIBLE PHONE LIGHT"
        alert_time = time.time()
        log("phone_light", score)

    if score >= MAX_SCORE:
        score = MAX_SCORE
        alert_text = "TERMINATED"
        save(frame, "terminated")
        log("terminated", score)

    if time.time() - alert_time < 2:
        cv2.putText(frame, alert_text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)

    cv2.putText(frame, "SCORE: " + str(score), (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)

    draw_box(frame, "ZONE 1 ACTIVE", (0, 255, 0), h - 60)
    draw_box(frame, "ZONE 2 ACTIVE", (255, 255, 0), h - 30)

    cv2.imshow("FINAL BOSS EXAM AI", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()