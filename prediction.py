"""
Parallel Prediction Script
Author: molingc1
Description: This script loads a 3D HDF5 dataset and processes each z-layer in parallel using multiple GPUs.
"""
# ================================================
# -*- coding: utf-8 -*-
# Requirements:
# - scaler.py               (contains PiecewiseScaler class)
# - worker.py               (defines worker_task function for GPU inference)
# - worker2.py              (defines process_z_layer function)
# - Pretrained model + scalers must be loaded within worker_task
# ================================================

import numpy as np
import pandas as pd
import time
import h5py
import multiprocessing as mp
import os
from worker import worker_task
from multiprocessing import shared_memory
from scaler import PiecewiseScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file = "grating012umpitch05dutycycle20um"
with h5py.File(file + ".h5", 'r') as f:
    dset = f[file]
    arr_3d_loaded = dset[()]

arr_3d_loaded = arr_3d_loaded[::1, ::1, ::1]
z_indices = list(range(arr_3d_loaded.shape[2]))
# z_indices = [110, 150, 200, 300]

if __name__ == '__main__':
    start_time = time.time()
    print("Start parallel processing on all z-layers...")

    if not mp.get_start_method(allow_none=True):
        mp.set_start_method('spawn', force=True)

    shm = shared_memory.SharedMemory(create=True, size=arr_3d_loaded.nbytes)
    shared_arr = np.ndarray(arr_3d_loaded.shape, dtype=arr_3d_loaded.dtype, buffer=shm.buf)
    shared_arr[:] = arr_3d_loaded[:]

    num_gpus = 2
    num_workers_per_gpu = 2
    total_workers = num_gpus * num_workers_per_gpu

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for z_idx in z_indices:
        if z_idx % 10 == 0:
            task_queue.put(z_idx)
    for _ in range(total_workers):
        task_queue.put(None)

    processes = []
    for i in range(total_workers):
        gpu_id = i % num_gpus
        p = mp.Process(target=worker_task, args=(task_queue, result_queue, shm.name, arr_3d_loaded.shape, arr_3d_loaded.dtype, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results_list = []
    while not result_queue.empty():
        results_list.append(result_queue.get())

    end_time = time.time()
    print(f"All z-layers processed. Total time: {end_time - start_time:.2f} s")

    shm.close()
    shm.unlink()

    results_list = [r for r in results_list if r is not None]
    # df_ape_2d = pd.DataFrame({
    #     'z_idx': [z_idx for z_idx, _ in results_list],
    #     'ape_1d': [ape_1d for _, ape_1d in results_list]
    # })

    # df_ape_2d.to_csv("APE_results.csv", index=False)
    # print("Saved results to APE_results.csv")
