import cv2
import time
import os
import subprocess
import re
import numpy as np
import torch
import optuna
from pathlib import Path
from ultralytics import RTDETR

try:
    from boxmot import BoTSORT
except ImportError:
    from boxmot import BotSort as BoTSORT

def extract_detections(results):
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 6))
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy().reshape(-1, 1)
    clss = r.boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.hstack((boxes, confs, clss))

def run_tracking_iteration(model, video_path, reid_weights, params):
    tracker = BoTSORT(
        model_weights=Path(reid_weights),
        device='cuda:0',
        fp16=True,
        track_high_thresh=params['track_high_thresh'],
        track_low_thresh=params['track_low_thresh'],
        new_track_thresh=params['new_track_thresh'],
        track_buffer=300, 
        match_thresh=params['match_thresh'],
        proximity_thresh=params['proximity_thresh'],
        appearance_thresh=0.05
    )
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_w = 1920.0 / width
    scale_h = 1080.0 / height

    mot_file_path = os.path.join("TrackEval", "data", "trackers", "mot_challenge", "MOT17-train", "L-MAT", "data", "MOT17-04-FRCNN.txt")
    mot_file = open(mot_file_path, "w")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame_count += 1
        results = model(frame, conf=params['conf'], imgsz=params['imgsz'], classes=[0], verbose=False)
        dets = extract_detections(results)
        tracks = tracker.update(dets, frame)
        
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, ind = track
            scaled_x1 = int(x1 * scale_w)
            scaled_y1 = int(y1 * scale_h)
            w = int(x2 * scale_w) - scaled_x1
            h = int(y2 * scale_h) - scaled_y1
            mot_file.write(f"{frame_count},{int(track_id)},{scaled_x1},{scaled_y1},{w},{h},{conf:.4f},-1,-1,-1\n")

    mot_file.close()
    cap.release()
    
    del tracker
    torch.cuda.empty_cache()

def evaluate_tracker():
    cmd = [
        "python", "scripts/run_mot_challenge.py", 
        "--BENCHMARK", "MOT17", 
        "--SPLIT_TO_EVAL", "train", 
        "--TRACKERS_TO_EVAL", "L-MAT", 
        "--METRICS", "HOTA", "CLEAR", "Identity", 
        "--USE_PARALLEL", "False", 
        "--NUM_PARALLEL_CORES", "1", 
        "--SEQ_INFO", "MOT17-04-FRCNN"
    ]
    result = subprocess.run(cmd, cwd="TrackEval", capture_output=True, text=True)
    output = result.stdout
    try:
        mota_match = re.search(r'CLEAR:.*?COMBINED\s+([-\d\.]+)', output, re.DOTALL)
        idf1_match = re.search(r'Identity:.*?COMBINED\s+([-\d\.]+)', output, re.DOTALL)
        return float(mota_match.group(1)) if mota_match else 0.0, float(idf1_match.group(1)) if idf1_match else 0.0
    except Exception:
        return 0.0, 0.0

def objective(trial):
    """Optuna will call this function repeatedly, intelligently guessing new parameters."""
    
    params = {
        'conf': trial.suggest_float('conf', 0.01, 0.31, step=0.03),
        'imgsz': trial.suggest_categorical('imgsz', [1088, 1440]),
        'track_high_thresh': trial.suggest_float('track_high_thresh', 0.05, 0.51, step=0.05),
        'track_low_thresh': trial.suggest_float('track_low_thresh', 0.01, 0.26, step=0.02),
        'new_track_thresh': trial.suggest_float('new_track_thresh', 0.15, 0.81, step=0.05),
        'match_thresh': trial.suggest_float('match_thresh', 0.50, 0.96, step=0.05),
        'proximity_thresh': trial.suggest_float('proximity_thresh', 0.60, 0.96, step=0.05)
    }

    print(f"\n=============================================")
    print(f"[*] Trial {trial.number} | Testing Optuna Params: {params}")

    video = "MOT17-04-SDP-raw.webm"
    reid = "osnet_ain_x1_0_msmt17.pt"

    run_tracking_iteration(global_model, video, reid, params)
    
    mota, idf1 = evaluate_tracker()
    print(f"[+] Trial {trial.number} Result -> MOTA: {mota} | IDF1: {idf1}")

    return mota

if __name__ == "__main__":
    weights = "rtdetr-l.pt"
    
    print(f"[*] Pre-loading RT-DETR Model into VRAM...")
    global_model = RTDETR(weights)

    study = optuna.create_study(
        study_name="MeMOT_Defeater", 
        storage="sqlite:///mot_study.db", 
        load_if_exists=True, 
        direction="maximize"
    )
    
    print("[*] Starting Bayesian Optimization Sequence...")
    study.optimize(objective, n_trials=80)

    print("\n=============================================")
    print("[+] BAYESIAN OPTIMIZATION COMPLETE!")
    print(f"[!] ABSOLUTE BEST SCORE (MOTA): {study.best_value}")
    print(f"[!] OPTIMAL MATHEMATICAL PARAMETERS:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")