import cv2
import time
import argparse
import os
import numpy as np
import glob
from pathlib import Path
from ultralytics import RTDETR

try:
    from boxmot import BoTSORT
except ImportError:
    from boxmot import BotSort as BoTSORT

def init_tracker(reid_weights):
    """Initializes the BoxMOT tracker with the heavy OSNet Re-ID model."""
    return BoTSORT(
        model_weights=Path(reid_weights),
        device='cuda:0',
        fp16=True,
        track_high_thresh=0.3458888874942636,
        track_low_thresh=0.04603513233314025,
        new_track_thresh=0.2928566501828457,
        track_buffer=300,
        match_thresh=0.8029549498576292,
        proximity_thresh=0.7517651056649984,
        appearance_thresh=0.8
    )

def extract_detections(results):
    """Formats Ultralytics RT-DETR detections into BoxMOT format [x1,y1,x2,y2,conf,cls]"""
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 6))
    
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy().reshape(-1, 1)
    clss = r.boxes.cls.cpu().numpy().reshape(-1, 1)
    return np.hstack((boxes, confs, clss))

def draw_tracks(frame, tracks):
    """Draws bounding boxes and IDs since we are no longer using Ultralytics r.plot()"""
    for track in tracks:
        x1, y1, x2, y2, track_id, conf, cls, ind = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 50, 50), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


def run_image_sequence(img_folder, weights, reid_weights):
    print(f"[*] Processing Image Sequence: {img_folder}")
    model = RTDETR(weights)
    tracker = init_tracker(reid_weights)
    
    images = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))
    if not images:
        print(f"[!] Error: No .jpg images found in {img_folder}")
        return
        
    sample = cv2.imread(images[0])
    height, width = sample.shape[:2]
    
    scale_w = 1920.0 / width
    scale_h = 1080.0 / height

    os.makedirs("results/images", exist_ok=True)
    mot_file_path = os.path.join("results", "mot_benchmark.txt")
    mot_file = open(mot_file_path, "w")

    frame_count = 0
    start_time = time.time()

    for img_path in images:
        frame = cv2.imread(img_path)
        frame_count += 1
        
        results = model(frame, conf=0.05, imgsz=1088, classes=[0], verbose=False)
        dets = extract_detections(results)
        tracks = tracker.update(dets, frame)
        
        annotated_frame = draw_tracks(frame.copy(), tracks)
        
        out_img_path = os.path.join("results", "images", f"tracked_{frame_count:04d}.jpg")
        cv2.imwrite(out_img_path, annotated_frame)
        
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, ind = track
            
            x1, y1 = int(x1 * scale_w), int(y1 * scale_h)
            x2, y2 = int(x2 * scale_w), int(y2 * scale_h)
            
            w, h = x2 - x1, y2 - y1
            mot_file.write(f"{frame_count},{int(track_id)},{x1},{y1},{w},{h},{conf:.4f},-1,-1,-1\n")

        if frame_count % 50 == 0:
            print(f"[*] Processed {frame_count} frames - FPS: {frame_count / (time.time() - start_time):.1f}")

    mot_file.close()
    print(f"\n[+] Sequence processing complete! Text saved to: {mot_file_path}")
    print(f"[+] Annotated images saved to: results/images/")


def run_video_file(source_video, weights, reid_weights):
    print(f"[*] Processing Video File: {source_video}")
    model = RTDETR(weights)
    tracker = init_tracker(reid_weights)
    
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        print(f"[!] Error: Could not open {source_video}")
        return

    os.makedirs("results", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(source_video))[0]
    output_path = os.path.join("results", f"tracked_{base_name}.mp4")
    
    mot_file_path = os.path.join("results", "mot_benchmark.txt")
    mot_file = open(mot_file_path, "w")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale_w = 1920.0 / width
    scale_h = 1080.0 / height

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        results = model(frame, conf=0.06784364415499451, imgsz=1088, classes=[0], verbose=False)
        dets = extract_detections(results)
        tracks = tracker.update(dets, frame)
        
        annotated_frame = draw_tracks(frame.copy(), tracks)

        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, ind = track
            
            scaled_x1 = int(x1 * scale_w)
            scaled_y1 = int(y1 * scale_h)
            scaled_x2 = int(x2 * scale_w)
            scaled_y2 = int(y2 * scale_h)
            
            w = scaled_x2 - scaled_x1
            h = scaled_y2 - scaled_y1
            mot_file.write(f"{frame_count},{int(track_id)},{scaled_x1},{scaled_y1},{w},{h},{conf:.4f},-1,-1,-1\n")

        live_fps = frame_count / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {live_fps:.1f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        out.write(annotated_frame)

        if frame_count % 50 == 0:
            print(f"[*] Processed {frame_count}/{total_frames} frames - Current FPS: {live_fps:.1f}")

    mot_file.close()
    cap.release()
    out.release()
    print(f"\n[+] Video processing complete! Saved to: {output_path}")
    print(f"[+] MOT Benchmark Data saved to: {mot_file_path}")


def run_cctv_stream(source_stream, weights, reid_weights):
    print(f"[*] Connecting to Live Stream: {source_stream}")
    model = RTDETR(weights)
    tracker = init_tracker(reid_weights)
    
    source = int(source_stream) if source_stream.isdigit() else source_stream
    cap = cv2.VideoCapture(source)
    
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        results = model(frame, conf=0.06784364415499451, imgsz=1088, classes=[0], verbose=False)
        dets = extract_detections(results)
        tracks = tracker.update(dets, frame)
        
        annotated_frame = draw_tracks(frame.copy(), tracks)

        live_fps = frame_count / (time.time() - start_time)
        cv2.putText(annotated_frame, f"Live FPS: {live_fps:.1f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Live CCTV Feed", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[*] Stream interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


def run_screen_capture(weights, reid_weights):
    import mss 
    print("[*] Starting Live Screen Capture...")
    model = RTDETR(weights)
    tracker = init_tracker(reid_weights)
    
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", f"screen_track_{int(time.time())}.mp4")

    sct = mss.mss()
    monitor = sct.monitors[1]
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (monitor["width"], monitor["height"]))
    cv2.namedWindow("Live Screen Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Screen Tracking", 1280, 720)
    
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            frame = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
            frame_count += 1

            results = model(frame, conf=0.06784364415499451, imgsz=1088, classes=[0], verbose=False)
            dets = extract_detections(results)
            tracks = tracker.update(dets, frame)
            
            annotated_frame = draw_tracks(frame.copy(), tracks)

            live_fps = frame_count / (time.time() - start_time)
            cv2.putText(annotated_frame, f"Live FPS: {live_fps:.1f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow("Live Screen Tracking", annotated_frame)
            out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        out.release()
        cv2.destroyAllWindows()
        print(f"[+] Screen capture complete! Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Real-Time BoxMOT Tracking CLI")
    parser.add_argument('--weights', type=str, default="rtdetr-l.pt", help="Path to RT-DETR weights")
    parser.add_argument('--reid', type=str, default="osnet_ain_x1_0_msmt17.pt", help="Path to Re-ID weights")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--screen', action='store_true', help="Track the main monitor live")
    group.add_argument('--cctv', type=str, help="Track a live stream (e.g., '0' for webcam or 'rtsp://link')")
    group.add_argument('--vid', type=str, help="Process a local video file (e.g., 'Times_Square.mp4')")
    group.add_argument('--imgdir', type=str, help="Process an official MOT image folder (e.g., 'MOT17-04/img1')")
    
    args = parser.parse_args()

    if args.screen:
        run_screen_capture(args.weights, args.reid)
    elif args.cctv:
        run_cctv_stream(args.cctv, args.weights, args.reid)
    elif args.vid:
        run_video_file(args.vid, args.weights, args.reid)
    elif args.imgdir:
        run_image_sequence(args.imgdir, args.weights, args.reid)
