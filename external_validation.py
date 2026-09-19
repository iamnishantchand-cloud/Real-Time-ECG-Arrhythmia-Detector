import os
import time
import json
import numpy as np
from scipy import signal
import wfdb
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def download_with_retry(record_name, pn_dir='incartdb', max_retries=3):
    for attempt in range(max_retries):
        try:
            record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
            annotation = wfdb.rdann(record_name, 'atr', pn_dir=pn_dir)
            return record, annotation
        except Exception as e:
            print(f"Error downloading {record_name}: {e}")
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt))
            else:
                return None, None

def map_annotation(symbol):
    n_class = ['N', '.', '/', 'L', 'R', 'e', 'j']
    s_class = ['A', 'a', 'J', 'S']
    v_class = ['V', 'E']
    f_class = ['F']
    q_class = ['Q', '?', 'U']
    
    if symbol in n_class: return 0
    if symbol in s_class: return 1
    if symbol in v_class: return 2
    if symbol in f_class: return 3
    if symbol in q_class: return 4
    return -1

def process_incart():
    records = [f'I{str(i).zfill(2)}' for i in range(1, 76)]
    
    X = []
    y = []
    
    for record_name in records:
        print(f"Processing {record_name}...")
        record, annotation = download_with_retry(record_name)
        if record is None:
            continue
        
        # Lead II is index 1 but we check if it is explicitly II
        lead_idx = 1
        if record.sig_name[lead_idx] != 'II':
            try:
                lead_idx = record.sig_name.index('II')
            except ValueError:
                print(f"Lead II not found in {record_name}")
                continue
                
        ecg_signal = record.p_signal[:, lead_idx]
        symbols = annotation.symbol
        sample_indices = annotation.sample
        
        # Beat window at 257 Hz (converted from 360Hz window of 100/149)
        pre = 71
        post = 106
        
        for i in range(len(symbols)):
            sym = symbols[i]
            label = map_annotation(sym)
            if label == -1:
                continue
                
            idx = sample_indices[i]
            if idx - pre < 0 or idx + post >= len(ecg_signal):
                continue
                
            beat_segment = ecg_signal[idx - pre : idx + post]
            
            # Resample to 250 samples
            resampled_beat = signal.resample(beat_segment, 250)
            
            # Normalize with z-score
            mean = np.mean(resampled_beat)
            std = np.std(resampled_beat)
            if std > 0:
                normalized_beat = (resampled_beat - mean) / std
            else:
                normalized_beat = resampled_beat - mean
                
            X.append(normalized_beat)
            y.append(label)
            
        time.sleep(1) # delay between downloads
        
    return np.array(X), np.array(y)

def main():
    os.makedirs('output', exist_ok=True)
    
    print("Extracting and processing INCART data...")
    X, y = process_incart()
    print(f"Processed {len(y)} beats.")
    
    # Reshape X for the model (N, 250, 1)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    print("Loading model...")
    model_path = 'output/ecg_model.keras'
    model = load_model(model_path, compile=False)
    
    print("Predicting...")
    y_pred_prob = model.predict(X)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    print("Computing metrics...")
    acc = accuracy_score(y, y_pred)
    cr_dict = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cr_str = classification_report(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    print("Classification Report:")
    print(cr_str)
    
    results = {
        'accuracy': acc,
        'classification_report': cr_dict,
        'confusion_matrix': cm.tolist()
    }
    
    with open('output/external_validation.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['N', 'S', 'V', 'F', 'Q'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix - INCART External Validation')
    plt.savefig('output/external_validation_cm.png')
    plt.close()
    
    print("Done!")

if __name__ == '__main__':
    main()
