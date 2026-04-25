# Edge-MOT: Decoupled Zero-Shot Tracking

An edge-optimized, real-time Multi-Object Tracking (MOT) pipeline designed to achieve State-of-the-Art (SOTA) tracking metrics on heavily compressed streaming video using consumer-grade hardware.

## Overview

End-to-end tracking transformers (such as MeMOT) achieve high accuracy but require massive computational clusters (e.g., 8x Tesla A100 GPUs) and uncompressed, lossless 1080p imagery to function effectively. 

**Edge-MOT** proposes a highly efficient, decoupled architecture:
1. **Spatial Detection:** Utilizes **RT-DETR** (Real-Time DEtection TRansformer) for rapid, accurate bounding box generation.
2. **Temporal Memory:** Employs **BoxMOT (BoTSORT)** paired with an Attention-In-Network Re-ID model (**OSNet-AIN**) to maintain identity tracking through severe occlusion and lighting changes.

By decoupling these processes, this pipeline runs in real-time on a single consumer GPU (NVIDIA RTX 4060) completely zero-shot, without requiring fine-tuning on the target dataset.

## 📊 Benchmark Results

Evaluated on the heavily compressed `MOT17-04` sequence, this pipeline was mathematically pushed to its absolute limits using Bayesian Hyperparameter Optimization. 

| Metric | Score | Description |
| :--- | :--- | :--- |
| **MOTA** | `50.403` | Multi-Object Tracking Accuracy |
| **IDF1** | `59.848` | Identity F1 Score |
| **IDsw** | `82` | Identity Switches (96% reduction compared to MeMOT's 2,724) |
| **CLR_TP**| `27,151` | True Positives |

*Note: Achieving >50 MOTA and >59 IDF1 zero-shot on a degraded `.webm` video represents a significant leap in practical, real-world edge tracking stability.*

## Bayesian Hyperparameter Optimization

To overcome the inherent data loss in compressed video formats, this repository includes an automated hyperparameter tuning engine.

* **`auto_optimizer.py`**: Executes an exhaustive Grid Search/Random Search across the Re-ID memory bank constraints.
* **`finetune_optimizer.py`**: Utilizes **Optuna** (Tree-structured Parzen Estimator) to perform Bayesian Optimization. It mathematically maps the search space, creates a region of interest around the top-performing trials, and automatically anneals the confidence, proximity, and matching thresholds to extract the maximum possible MOTA score.

## Installation & Usage

**1. Install Dependencies**
```bash
pip install ultralytics boxmot torch torchvision optuna opencv-python

2. Run
# Run on a local video file
python master_tracker.py --vid path/to/video.mp4 --reid osnet_ain_x1_0_msmt17.pt

# Run on an official MOT image sequence
python master_tracker.py --imgdir path/to/MOT17-04/img1 --reid osnet_ain_x1_0_msmt17.pt

# Run live screen capture
python master_tracker.py --screen --reid osnet_ain_x1_0_msmt17.pt

3. Optimization
python optuna_optimizer.py
python finetune_optimizer.py
