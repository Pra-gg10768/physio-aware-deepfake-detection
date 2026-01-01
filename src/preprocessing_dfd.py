import os
from tqdm import tqdm
from src.preprocessing import process_video

def run_dfd_preprocessing(
    raw_root="data/dfd_original_sequence",
    out_root="data/signals_dfd_test"
):
    print("[INFO] DFD preprocessing started")

    label = "unknown"  

    videos = [v for v in os.listdir(raw_root) if v.endswith(".mp4")]

    for video in tqdm(videos, desc="DFD videos"):
        video_path = os.path.join(raw_root, video)
        process_video(video_path, out_root, label)


if __name__ == "__main__":
    run_dfd_preprocessing()
