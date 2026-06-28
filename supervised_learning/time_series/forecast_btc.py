#!/usr/bin/env python3
"""Train and evaluate an RNN for BTC price forecasting."""
import sys

import numpy as np
import pandas as pd
import tensorflow as tf


WINDOW_SIZE = 24
BATCH_SIZE = 32
SHUFFLE_BUFFER = 1024
DEFAULT_DATA_PATH = 'btc_hourly.csv'


def _build_windows(features, targets, window_size=WINDOW_SIZE):
    """Convert arrays into sliding windows for sequence learning."""
    sequences = []
    labels = []
    limit = len(features) - window_size
    for index in range(limit):
        sequences.append(features[index:index + window_size])
        labels.append(targets[index + window_size])
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def _make_dataset(features, targets, batch_size=BATCH_SIZE, shuffle=False):
    """Create a tf.data.Dataset from feature and target arrays."""
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    if shuffle:
        dataset = dataset.shuffle(min(len(features), SHUFFLE_BUFFER))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def _scale_with_training_statistics(train_frame, full_frame):
    """Scale all columns using training-set mean and standard deviation."""
    mean = train_frame.mean(axis=0)
    std = train_frame.std(axis=0).replace(0, 1)
    return (full_frame - mean) / std, mean, std


def load_hourly_data(path=DEFAULT_DATA_PATH):
    """Load the preprocessed hourly BTC dataset."""
    frame = pd.read_csv(path, parse_dates=['timestamp'])
    frame = frame.sort_values('timestamp').reset_index(drop=True)
    feature_frame = frame.drop(columns=['timestamp'])
    return frame, feature_frame


def build_model(input_shape):
    """Build a recurrent model for BTC forecasting."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_forecaster(data_path=DEFAULT_DATA_PATH, epochs=25):
    """Train and evaluate the forecasting model."""
    frame, feature_frame = load_hourly_data(data_path)
    if len(feature_frame) <= WINDOW_SIZE + 1:
        raise ValueError('not enough hourly rows to build windows')

    train_end = int(len(feature_frame) * 0.7)
    val_end = int(len(feature_frame) * 0.85)

    scaled_features, mean, std = _scale_with_training_statistics(
        feature_frame.iloc[:train_end], feature_frame)
    target_column = 'close'
    scaled_targets = (feature_frame[target_column] - mean[target_column]) / std[target_column]

    x_all, y_all = _build_windows(scaled_features.values, scaled_targets.values)
    target_offset = WINDOW_SIZE
    train_limit = max(train_end - target_offset, 0)
    val_limit = max(val_end - target_offset, 0)

    x_train = x_all[:train_limit]
    y_train = y_all[:train_limit]
    x_val = x_all[train_limit:val_limit]
    y_val = y_all[train_limit:val_limit]
    x_test = x_all[val_limit:]
    y_test = y_all[val_limit:]

    train_ds = _make_dataset(x_train, y_train, shuffle=True)
    val_ds = _make_dataset(x_val, y_val)
    test_ds = _make_dataset(x_test, y_test)

    model = build_model((WINDOW_SIZE, x_all.shape[-1]))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_mae = model.evaluate(test_ds, verbose=0)
    predictions = model.predict(test_ds)
    predictions = predictions.reshape(-1)
    actual = y_test.reshape(-1)

    results = {
        'model': model,
        'history': history.history,
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'predictions': predictions,
        'actual': actual,
        'feature_mean': mean,
        'feature_std': std
    }
    return results


def main(argv=None):
    """Train the forecaster from the command line."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (0, 1):
        raise SystemExit('Usage: ./forecast_btc.py [btc_hourly.csv]')

    data_path = args[0] if args else DEFAULT_DATA_PATH
    results = train_forecaster(data_path)
    print('Test loss: {:.6f}'.format(results['test_loss']))
    print('Test MAE: {:.6f}'.format(results['test_mae']))


if __name__ == '__main__':
    main()
