"""
Z-layer Prediction Processor
Author: molingc1

Description:
Core function to process one z-index slice in a 3D dataset:
- Extract features
- Predict using Transformer model
- Update history
- Compute APE
- Save plots and CSVs
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import os
import tensorflow as tf

def process_z_layer(z_idx, shared_arr, model, scaler_x, scaler_y, feature_num):
    """ Process a single Z-slice of electric field data. """
    import time
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pandas_ta as ta
    import os
    import tensorflow as tf

    print(f'z{z_idx} prediction started')
    z_slice = shared_arr[:, :, z_idx]
    df_y_2d = z_slice.transpose()
    df = pd.DataFrame(df_y_2d)

    df_data_list = []
    for i in range(df.shape[0]):
        series_e = df.iloc[i, :]
        df_data = pd.DataFrame()
        df_data['e'] = series_e
        # df_data['after1'] = df_data['e'].shift(-10)
        # df_data['after2'] = df_data['e'].shift(-20)
        df_data['e_diff10'] = df_data['e'] - df_data['e'].shift(-10)
        df_data['e_diff20'] = df_data['e'] - df_data['e'].shift(-20)
        df_data['x'] = range(df.shape[1])
        df_data['RSI'] = ta.rsi(df_data['e'], length=30)
        df_data['EMAS'] = ta.ema(df_data['e'], length=10)
        df_data['EMAM'] = ta.ema(df_data['e'], length=20)
        df_data['EMAL'] = ta.ema(df_data['e'], length=30)
        for j in range(1, 30):
            df_data[f'next{j}'] = df_data['e'].shift(-j)
        df_data.dropna(inplace=True)
        df_data.reset_index(drop=True, inplace=True)
        df_data_list.append(df_data)

    n_rows = len(df_data_list)
    n_time = df_data_list[0].shape[0]
    pred_indices = np.arange(40, n_time, 30)

    alt_list = []
    for i in range(n_rows):
        initial_df = df_data_list[i].iloc[:40, :feature_num].copy()
        alt_rows = initial_df.to_dict(orient='records')
        alt_list.append(alt_rows)
    columns_order = df_data_list[0].iloc[:40, :feature_num].columns.tolist()

    for idx in pred_indices:
        batch_features_denorm = []
        for i in range(n_rows):
            current_row = df_data_list[i].iloc[idx, :feature_num].to_dict()
            alt_list[i].append(current_row)
            e_series = pd.Series([row['e'] for row in alt_list[i]])
            rsi_series = ta.rsi(e_series, length=30)
            emas_series = ta.ema(e_series, length=10)
            emam_series = ta.ema(e_series, length=20)
            emal_series = ta.ema(e_series, length=30)
            alt_list[i][-1]['RSI'] = rsi_series.iloc[-1]
            alt_list[i][-1]['EMAS'] = emas_series.iloc[-1]
            alt_list[i][-1]['EMAM'] = emam_series.iloc[-1]
            alt_list[i][-1]['EMAL'] = emal_series.iloc[-1]
            feature_vector = [alt_list[i][-1].get(col, np.nan) for col in columns_order]
            batch_features_denorm.append(feature_vector)

        # print('YJ transform')
        feature_cols = ['e','e_diff10','e_diff20','x','RSI','EMAS','EMAM','EMAL']
        batch_features_denorm = pd.DataFrame(batch_features_denorm, columns=feature_cols)
        batch_features = {}
        for col in feature_cols:
            data = batch_features_denorm[col].values.astype(float)
            scaler = scaler_x[col]
            if col == 'e':
                batch_features[col] = scaler.transform(data)
            else:
                batch_features[col] = scaler.transform(data.reshape(-1, 1)).flatten()

        batch_features_array = np.stack([batch_features[col] for col in feature_cols], axis=1)
        batch_input = batch_features_array.reshape(batch_features_array.shape[0], batch_features_array.shape[1], 1)
        y_pred_batch = model.predict(batch_input, verbose=0)
        y_pred_batch = y_pred_batch.reshape(y_pred_batch.shape[0], y_pred_batch.shape[1])

        def inv_transform_labels(y_transformed):
            y_original = np.empty_like(y_transformed)
            n_labels = y_transformed.shape[1]
            for i in range(n_labels):
                col_data = y_transformed[:, i].reshape(-1, 1)
                y_original[:, i] = scaler_y[i].inverse_transform(col_data).flatten()
            return y_original

        y_pred_batch_denorm = inv_transform_labels(y_pred_batch)

        for i in range(n_rows):
            for j in range(29):
                new_row = {}
                new_row[columns_order[0]] = y_pred_batch_denorm[i, j]
                for k, col in enumerate(columns_order):
                    if k == 0:
                        continue
                    elif k == 7:
                        new_row[col] = idx + j + 1
                    else:
                        new_row[col] = np.nan
                alt_list[i].append(new_row)

    output_folder = os.path.join("results", str(z_idx))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    selected_ape_entries = []
    selected_y_indices = range(0, 975, 30)

    for i in range(n_rows):
        if i in selected_y_indices:
            alt_df = pd.DataFrame(alt_list[i])
            length = df_data_list[i].shape[0]
            e_predicted = alt_df['e'].iloc[:length]
            e_actual = df_data_list[i]['e']
            real_indices = list(range(40)) + list(range(40, len(e_predicted), 30))
            all_indices = np.arange(len(e_predicted))
            mask = np.isin(all_indices, real_indices, invert=True)

            plt.figure(figsize=(10, 5))
            plt.scatter(all_indices[mask], e_predicted[mask], s=2, color='blue', label='Predicted')
            plt.scatter(all_indices[real_indices], e_predicted[real_indices], s=5, color='red', label='Real Values')
            plt.plot(e_actual, color='black', linewidth=1.5, alpha=0.7, label='Actual')
            plt.title(f"y = {i}")
            x_ticks = np.linspace(-20, 80, 6)
            x_ticks_positions = np.linspace(0, 1950, 6)
            plt.xticks(x_ticks_positions, x_ticks)
            plt.xlabel("X (μm)")
            plt.ylabel("Electric Field (eV)")
            plt.legend()
            plt.savefig(os.path.join(output_folder, f"comparison_y_{i}.png"))
            plt.close()

            csv_path = os.path.join(output_folder, 'e_fields')
            os.makedirs(csv_path, exist_ok=True)
            e_predicted.to_csv(os.path.join(csv_path, f"e_predicted_{i}.csv"), index=False)

            ape_1d = np.mean(np.abs((e_actual - e_predicted) / e_predicted) * 100)
            selected_ape_entries.append((i, ape_1d))

    ape_2d = np.mean([entry[1] for entry in selected_ape_entries]) if selected_ape_entries else None
    print(f"Z slice overall APE: {ape_2d}")

    csv_filename = os.path.join(output_folder, "APE_results.csv")
    file_exists = os.path.exists(csv_filename)
    df_entry = pd.DataFrame(selected_ape_entries, columns=["y_idx", "ape_1d"])
    df_entry.to_csv(csv_filename, mode='a', header=not file_exists, index=False)

    tf.keras.backend.clear_session()
    return {
        'z_idx': z_idx,
        'ape_2d': ape_2d,
        'ape_1d_list': [entry[1] for entry in selected_ape_entries],
        'output_folder': output_folder
    }
