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
        
        results = model(frame, conf=params['conf'], imgsz=1088, classes=[0], verbose=False)
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
        "python", "scripts/run_mot_challenge.py", "--BENCHMARK", "MOT17", 
        "--SPLIT_TO_EVAL", "train", "--TRACKERS_TO_EVAL", "L-MAT", 
        "--METRICS", "HOTA", "CLEAR", "Identity", "--USE_PARALLEL", "False", 
        "--NUM_PARALLEL_CORES", "1", "--SEQ_INFO", "MOT17-04-FRCNN"
    ]
    result = subprocess.run(cmd, cwd="TrackEval", capture_output=True, text=True)
    try:
        mota = float(re.search(r'CLEAR:.*?COMBINED\s+([-\d\.]+)', result.stdout, re.DOTALL).group(1))
        idf1 = float(re.search(r'Identity:.*?COMBINED\s+([-\d\.]+)', result.stdout, re.DOTALL).group(1))
        return mota, idf1
    except Exception:
        return 0.0, 0.0

def extract_top_20_boundaries(db_path, study_name):
    """Reads the old Optuna database and calculates the mathematical ROI limits."""
    print(f"[*] Reading historical data from {db_path}...")
    old_study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
    
    completed = [t for t in old_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_20 = sorted(completed, key=lambda x: x.value, reverse=True)[:20]
    
    print(f"[+] Found Top 20 Trials. Score Range: {top_20[-1].value} -> {top_20[0].value}")
    
    param_lists = {k: [] for k in top_20[0].params.keys() if k != 'imgsz'}
    
    for trial in top_20:
        for key in param_lists.keys():
            param_lists[key].append(trial.params[key])
            
    bounds = {}
    for key, vals in param_lists.items():
        min_val = max(0.001, min(vals) - 0.01) 
        max_val = min(1.0, max(vals) + 0.01)   
        bounds[key] = (min_val, max_val)
        print(f"    -> {key} optimal range: [{min_val:.4f} to {max_val:.4f}]")
        
    return bounds

def finetune_objective(trial):
    """Uses the dynamically calculated bounds to run a highly targeted search."""
    params = {}
    for key, (min_val, max_val) in optimal_bounds.items():
        params[key] = trial.suggest_float(key, min_val, max_val)

    print(f"\n=============================================")
    print(f"[*] Fine-Tune Trial {trial.number} | Params: {params}")

    run_tracking_iteration(global_model, "MOT17-04-SDP-raw.webm", "osnet_ain_x1_0_msmt17.pt", params)
    mota, idf1 = evaluate_tracker()
    
    print(f"[+] Fine-Tune Trial {trial.number} Result -> MOTA: {mota} | IDF1: {idf1}")
    return mota

if __name__ == "__main__":
    print(f"[*] Pre-loading RT-DETR Model into VRAM...")
    global_model = RTDETR("rtdetr-l.pt")
    
    optimal_bounds = extract_top_20_boundaries("mot_study.db", "MeMOT_Defeater")
    
    finetune_study = optuna.create_study(
        study_name="MeMOT_Finetune", 
        storage="sqlite:///mot_finetune.db", 
        load_if_exists=True, 
        direction="maximize"
    )
    
    print("\n[*] Starting Micro-Targeted Fine-Tuning Sequence...")
    finetune_study.optimize(finetune_objective, n_trials=200)

    print("\n=============================================")
    print("[+] FINE-TUNING COMPLETE!")
    print(f"[!] ABSOLUTE BEST SCORE (MOTA): {finetune_study.best_value}")
    for key, value in finetune_study.best_params.items():
        print(f"    {key}: {value}")