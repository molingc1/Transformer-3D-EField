# Transformer-3D-EField

A simplified Transformer-based model for fast and accurate 3D electric field prediction.

---

## 📌 Overview

This repository contains the full implementation of a deep learning approach that leverages a Transformer architecture to predict 3D electric fields from structured grating data. The method achieves high accuracy (up to 98% at focal regions) while significantly reducing simulation time and offering adjustable performance-memory trade-offs.

---

## 🧠 Key Features

- Transformer model with multi-head self-attention for spatial sequence modeling  
- Yeo-Johnson-based piecewise normalization for wide dynamic range  
- Custom relative error visualization callback  
- Parallelized z-layer prediction using shared memory and multi-GPU inference  
- Auto-APE (Absolute Percentage Error) visualization and statistics for evaluation

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `model_training.py` | Data preprocessing, normalization, model definition, training pipeline, and loss tracking |
| `prediction.py`    | Parallel inference across z-layers with shared memory, multiprocessing, and GPU control |
| `scaler.py`        | Custom `PiecewiseScaler` using Yeo-Johnson for feature/label transformation |
| `worker.py`        | Worker definition for per-z-slice GPU inference in multiprocessing setup |
| `worker2.py`       | Core prediction and APE visualization logic for each z-slice |

---

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/molingc1/Transformer-3D-EField.git
cd Transformer-3D-EField
