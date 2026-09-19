import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json
import os
import scipy.signal as signal

# Config
BASE_DIR = r"C:\Users\DELL\ecg-arrhythmia-detector"
MODEL_PATH = os.path.join(BASE_DIR, "output", "ecg_model.keras")
X_TEST_PATH = os.path.join(BASE_DIR, "output", "X_test.npy")
Y_TEST_PATH = os.path.join(BASE_DIR, "output", "y_test.npy")
OUTPUT_PLOT = os.path.join(BASE_DIR, "output", "noise_robustness_curve.png")
OUTPUT_JSON = os.path.join(BASE_DIR, "output", "noise_robustness.json")

# Noise levels (Clean represented as None)
SNRs = [None, 30, 20, 10, 5, 0, -5]

def calculate_noise_scale(signal, target_snr_db, noise):
    # calculate signal power
    signal_power = np.mean(signal**2, axis=-1, keepdims=True)
    # calculate noise power
    noise_power = np.mean(noise**2, axis=-1, keepdims=True)
    
    # Avoid div by 0
    noise_power = np.where(noise_power == 0, 1e-10, noise_power)
    scale = np.sqrt((signal_power / noise_power) * (10 ** (-target_snr_db / 10)))
    return scale

def add_gaussian_noise(x, snr):
    noise = np.random.normal(0, 1, size=x.shape)
    scale = calculate_noise_scale(x, snr, noise)
    return x + noise * scale

def add_baseline_wander(x, snr):
    # length 250, assume fs=125, t=0 to 2
    t = np.linspace(0, 2, x.shape[-1])
    # 0.5 Hz sine wave
    noise = np.sin(2 * np.pi * 0.5 * t)
    # broadcast to batch
    noise = np.tile(noise, (x.shape[0], 1))
    scale = calculate_noise_scale(x, snr, noise)
    return x + noise * scale

def add_muscle_artifact(x, snr):
    # band-limited noise 5-100Hz
    # generate white noise
    white_noise = np.random.normal(0, 1, size=x.shape)
    # apply filter (assume fs=250 for nyquist=125)
    # 5-100Hz is 5/125 to 100/125
    b, a = signal.butter(4, [5/125.0, 100/125.0], btype='bandpass')
    noise = signal.filtfilt(b, a, white_noise, axis=-1)
    scale = calculate_noise_scale(x, snr, noise)
    return x + noise * scale

def add_combined_noise(x, snr):
    n1 = np.random.normal(0, 1, size=x.shape)
    
    t = np.linspace(0, 2, x.shape[-1])
    n2 = np.tile(np.sin(2 * np.pi * 0.5 * t), (x.shape[0], 1))
    
    wn = np.random.normal(0, 1, size=x.shape)
    b, a = signal.butter(4, [5/125.0, 100/125.0], btype='bandpass')
    n3 = signal.filtfilt(b, a, wn, axis=-1)
    
    noise = n1 + n2 + n3
    scale = calculate_noise_scale(x, snr, noise)
    return x + noise * scale

def corrupt_data(x, noise_type, snr):
    if snr is None:
        return x
    if noise_type == "Gaussian":
        return add_gaussian_noise(x, snr)
    elif noise_type == "Baseline Wander":
        return add_baseline_wander(x, snr)
    elif noise_type == "Muscle Artifact":
        return add_muscle_artifact(x, snr)
    elif noise_type == "Combined":
        return add_combined_noise(x, snr)
    return x

def main():
    print("Loading data and model...")
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test
        
    model = load_model(MODEL_PATH, compile=False)
    
    noise_types = ["Gaussian", "Baseline Wander", "Muscle Artifact", "Combined"]
    
    results = {nt: {} for nt in noise_types}
    
    for nt in noise_types:
        print(f"\nEvaluating Noise Type: {nt}")
        for snr in SNRs:
            snr_label = "Clean" if snr is None else f"{snr}dB"
            
            # Corrupt data
            x_corr = corrupt_data(X_test, nt, snr)
            
            # Reshape for model
            x_corr = np.expand_dims(x_corr, axis=-1)
            
            # Predict
            preds = model.predict(x_corr, batch_size=512, verbose=0)
            if len(preds.shape) > 1 and preds.shape[1] > 1:
                pred_labels = np.argmax(preds, axis=1)
            else:
                pred_labels = (preds > 0.5).astype(int).flatten()
                
            acc = accuracy_score(y_test_labels, pred_labels)
            results[nt][snr_label] = acc
            print(f"  SNR: {snr_label:5s} | Accuracy: {acc:.4f}")
            
    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)
        
    # Plotting
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    x_labels = ["Clean", "30dB", "20dB", "10dB", "5dB", "0dB", "-5dB"]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, nt in enumerate(noise_types):
        accs = [results[nt][lbl] for lbl in x_labels]
        plt.plot(x_labels, accs, marker='o', label=nt, color=colors[i], linewidth=2)
        
    plt.axhline(y=results["Gaussian"]["Clean"], color='w', linestyle='--', alpha=0.5, label='Baseline (Clean)')
    
    plt.title("Model Robustness to Various Noise Types", fontsize=14, fontweight='bold')
    plt.xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"\nPlot saved to {OUTPUT_PLOT}")
    print(f"Results saved to {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
