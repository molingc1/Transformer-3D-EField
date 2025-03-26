"""
Worker Script for Parallel Prediction (Per z-layer)
Author: molingc1

Description:
- This script defines worker_task() function used in multiprocessing.
- Each worker:
    * connects to shared memory,
    * sets its own GPU environment,
    * loads model and scalers,
    * processes a single z-index using process_z_layer().
"""

import os
import multiprocessing as mp
from worker2 import process_z_layer
import numpy as np
from multiprocessing import shared_memory
import time
from scaler import PiecewiseScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model_path = "model_YJ.keras"

def worker_task(q, results, shared_data_name, shape, dtype, gpu_id):
    # Connect to shared memory
    existing_shm = shared_memory.SharedMemory(name=shared_data_name)
    shared_arr = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from tensorflow.keras.models import load_model
    model = load_model(model_path)

    import joblib
    scaler_x = joblib.load('scalers_features.pkl')
    scaler_y = joblib.load('scalers_labels.pkl')

    feature_num = 8

    while True:
        z_idx = q.get()
        if z_idx is None:
            break
        result = process_z_layer(z_idx, shared_arr, model, scaler_x, scaler_y, feature_num)
        results.put(result)

    existing_shm.close()
