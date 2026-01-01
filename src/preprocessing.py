import cv2
import os
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# ---------------------------
# MediaPipe FaceMesh
# ---------------------------

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Stable landmark ROIs
ROI_GROUPS = {
    "face": None,  # full face bbox
    "forehead": [10, 67, 69, 104, 108, 109, 151],
    "left_cheek": [234, 93, 132, 58, 172],
    "right_cheek": [454, 323, 361, 288, 397],
    "nose": [1, 2, 98, 327],
    "chin": [152, 175, 199]
}

ROI_NAMES = list(ROI_GROUPS.keys())


# ---------------------------
# Helpers
# ---------------------------

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    return res.multi_face_landmarks[0]


def roi_mean_rgb(frame, pts):
    x, y, w, h = cv2.boundingRect(pts)
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    return roi.mean(axis=(0, 1))  # (R,G,B)


# ---------------------------
# VIDEO → SIGNAL
# ---------------------------

def process_video(video_path, out_root, label):
    cap = cv2.VideoCapture(video_path)

    roi_signals = {k: [] for k in ROI_NAMES}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame)
        if landmarks is None:
            continue

        h, w, _ = frame.shape

        # Face bbox (using all landmarks)
        pts_all = np.array([
            (int(lm.x * w), int(lm.y * h))
            for lm in landmarks.landmark
        ])

        face_rgb = roi_mean_rgb(frame, pts_all)
        if face_rgb is not None:
            roi_signals["face"].append(face_rgb)

        # Sub-ROIs
        for roi, idxs in ROI_GROUPS.items():
            if idxs is None:
                continue

            pts = np.array([
                (int(landmarks.landmark[i].x * w),
                 int(landmarks.landmark[i].y * h))
                for i in idxs
            ])

            rgb = roi_mean_rgb(frame, pts)
            if rgb is not None:
                roi_signals[roi].append(rgb)

    cap.release()

    # ---------------------------
    # Temporal alignment
    # ---------------------------
    min_len = min(len(v) for v in roi_signals.values())
    if min_len < 60:
        return  # discard short videos

    signal = np.stack([
        np.array(roi_signals[k][:min_len])
        for k in ROI_NAMES
    ], axis=1)  # (T, ROIs, RGB)

    os.makedirs(os.path.join(out_root, label), exist_ok=True)
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    np.save(os.path.join(out_root, label, f"{video_id}.npy"), signal)


# ---------------------------
# MAIN PIPELINE
# ---------------------------

def run_preprocessing(
    raw_root="data/raw_videos",
    out_root="data/signals"
):
    print("[INFO] Preprocessing started")

    for label in ["real", "fake"]:
        label_dir = os.path.join(raw_root, label)
        if not os.path.exists(label_dir):
            continue

        for dataset in os.listdir(label_dir):
            dataset_dir = os.path.join(label_dir, dataset)
            if not os.path.isdir(dataset_dir):
                continue

            videos = [v for v in os.listdir(dataset_dir) if v.endswith(".mp4")]

            for video in tqdm(videos, desc=f"{label}/{dataset}"):
                video_path = os.path.join(dataset_dir, video)
                process_video(video_path, out_root, label)


