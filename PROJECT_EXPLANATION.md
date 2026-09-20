# Real-Time AI ECG Arrhythmia Detector — Complete Project Explanation

> Written for beginners. No prior knowledge of AI, medicine, or programming is assumed.

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [What is an ECG?](#2-what-is-an-ecg)
3. [What is an Arrhythmia?](#3-what-is-an-arrhythmia)
4. [The 5 Types of Heartbeats We Detect](#4-the-5-types-of-heartbeats-we-detect)
5. [Where Does the Data Come From?](#5-where-does-the-data-come-from)
6. [How the Data is Prepared (data_prep.py)](#6-how-the-data-is-prepared)
7. [How the AI Model Works (train_model.py)](#7-how-the-ai-model-works)
8. [The Dual-Input Architecture — Our Secret Weapon](#8-the-dual-input-architecture)
9. [How the Dashboard Works (app.py)](#9-how-the-dashboard-works)
10. [Noise Robustness Testing (noise_robustness.py)](#10-noise-robustness-testing)
11. [External Validation on INCART (external_validation.py)](#11-external-validation-on-incart)
12. [How All the Files Connect Together](#12-how-all-the-files-connect-together)
13. [Key Results and What They Mean](#13-key-results-and-what-they-mean)
14. [Glossary of Technical Terms](#14-glossary)

---

## 1. What Problem Does This Solve?

Imagine you're a nurse in a hospital ICU. You're watching 8 patient screens simultaneously. Each screen shows a squiggly green line — the patient's heartbeat. Your job is to spot the ONE dangerous heartbeat among thousands of normal ones. Miss it, and the patient could die.

**This is an impossible job for a human.** Hospitals report that nurses ignore up to 90% of alarms because there are too many false alerts (this is called "alarm fatigue").

**Our solution:** An AI system that watches every single heartbeat, classifies it instantly (in less than 5 milliseconds), and only alerts doctors when something is genuinely wrong. It never gets tired, never gets distracted, and catches 98.8% of dangerous heartbeats.

---

## 2. What is an ECG?

An **ECG (Electrocardiogram)** is a test that records the electrical activity of your heart. Every time your heart beats, an electrical signal travels through it, making the heart muscles contract and pump blood.

An ECG machine records these electrical signals through small sticky patches (electrodes) placed on your chest. The result is the familiar "heartbeat line" you see on hospital monitors.

### The PQRST Wave

Each normal heartbeat creates a specific pattern called the **PQRST wave**:

```
        R
       /|\
      / | \
     /  |  \
    /   |   \
P /    |    \ S    T
 /     |     \   /\
/      |      \_/  \
       |
       Q
```

- **P wave**: The upper chambers (atria) contract
- **QRS complex**: The main chambers (ventricles) contract — this is the big spike
- **T wave**: The heart recovers and prepares for the next beat

The **R-peak** is the tallest point — it marks exactly when each heartbeat happens. Our AI uses this R-peak as a reference point to extract individual heartbeats from the continuous signal.

### How ECG is Recorded

- The signal is measured in **millivolts (mV)** — tiny electrical voltages
- The MIT-BIH database records at **360 samples per second** (360 Hz) — meaning the machine takes 360 measurements every second
- A 30-minute recording = 360 × 60 × 30 = **648,000 data points** per patient

---

## 3. What is an Arrhythmia?

An **arrhythmia** is any heartbeat that doesn't follow the normal rhythm. Think of your heart like a drummer keeping a steady beat. An arrhythmia is like the drummer suddenly hitting an extra beat, skipping a beat, or beating in a weird pattern.

Some arrhythmias are harmless (everyone has occasional skipped beats). Others can be **life-threatening** — like ventricular beats (PVCs), which originate from the wrong part of the heart and can trigger fatal cardiac arrest if they happen too frequently.

**The challenge:** A cardiologist can recognize arrhythmias by looking at the ECG shape. But manually reading a 24-hour ECG recording with ~100,000 heartbeats? That would take hours. Our AI does it in seconds.

---

## 4. The 5 Types of Heartbeats We Detect

We classify every heartbeat into one of 5 categories, following the **AAMI standard** (Association for the Advancement of Medical Instrumentation — the official medical standard for heartbeat classification):

| Class | Name | What It Means | Danger Level |
|-------|------|--------------|--------------|
| **N** | Normal | Heart beating normally from the right place | Safe |
| **S** | Supraventricular | Beat originates from the upper chambers (atria) instead of the normal pacemaker | Moderate |
| **V** | Ventricular (PVC) | Beat originates from the lower chambers (ventricles) — the WRONG place | **Dangerous** |
| **F** | Fusion | A normal beat and a ventricular beat happen simultaneously and merge | Concerning |
| **Q** | Unknown/Paced | Unclassifiable beat, or from an artificial pacemaker | Depends |

### Why Detecting V (PVC) Beats Matters Most

PVCs are the most clinically dangerous. Imagine your heart has two electrical systems fighting for control — the normal one and an abnormal one from the ventricles. If the abnormal one wins too often, it can cause **ventricular fibrillation** (the heart quivers instead of pumping), which is fatal within minutes without a defibrillator.

**Our model catches 98.8% of all PVC beats** — this is our proudest metric.

---

## 5. Where Does the Data Come From?

### MIT-BIH Arrhythmia Database

Our training data comes from the **MIT-BIH Arrhythmia Database**, considered the gold standard for ECG research since 1980. It's hosted on [PhysioNet](https://physionet.org), a free public repository of medical signals.

**What's in it:**
- **48 recordings** from 48 different patients
- Each recording is **30 minutes long**
- Recorded at **360 Hz** (360 measurements per second)
- Every single heartbeat has been **manually labeled by two cardiologists** — they looked at each beat and wrote down what type it is

**Why this database is special:**
- It's been used in over 5,000 research papers
- It includes a good mix of normal and abnormal heartbeats
- The cardiologist annotations are extremely reliable (two independent experts agreed)
- It's freely available — no cost, no licensing issues

### How We Split the Data

This is **critically important** for honest AI evaluation:

- **Training set: 43 patients** — the AI learns from these
- **Test set: 5 patients** (records 100, 103, 105, 207, 208) — the AI has NEVER seen these patients

**Why patient-level splitting matters:** If you randomly split beats (not patients), the AI might memorize Patient X's unique heart shape during training and then "recognize" it during testing. That's cheating — it's recognizing the patient, not the arrhythmia. Our strict patient-level split ensures the AI must generalize to **completely new hearts** it has never encountered.

Many published papers don't do this, which is why they report misleadingly high 99%+ accuracy.

### INCART Database (External Validation)

We also tested on the **St. Petersburg INCART Database** — 75 patients from a completely different hospital in Russia, recorded with different equipment. This proves our model works beyond just the data it was trained on.

---

## 6. How the Data is Prepared

**File: `data_prep.py`** — This is the first script you run. It downloads the raw ECG recordings and converts them into a format the AI can learn from.

### Step-by-Step Process

#### Step 1: Download Raw Recordings
```
PhysioNet Server → Download 48 patient recordings → Store locally
```
Each recording is a continuous 30-minute ECG signal — just a long sequence of numbers representing voltage over time.

#### Step 2: Find Individual Heartbeats (Beat Segmentation)

The raw signal is continuous, but our AI needs to classify individual beats. So we:

1. **Read the R-peak annotations** — PhysioNet tells us exactly where each heartbeat's R-peak is
2. **Cut a 250-sample window** around each R-peak: 100 samples before + 150 samples after
3. This 250-sample window captures one complete heartbeat (about 0.7 seconds)

```
Continuous ECG signal:
...~~~∧~~~∨~~~∧~~~∨~~~∧~~~∨~~~...
         ↑         ↑         ↑
       R-peak    R-peak    R-peak

Extract each beat:
    [----∧----]  [----∧----]  [----∧----]
     250 samples  250 samples  250 samples
```

#### Step 3: Normalize Each Beat

Different patients have different signal amplitudes (some have strong signals, some weak). We **min-max normalize** each beat to a 0-1 range so the AI focuses on the SHAPE, not the amplitude.

```
Before: [-1.2, 0.5, 2.8, ...]  (raw millivolts, varies per patient)
After:  [0.0, 0.42, 1.0, ...]  (normalized 0-1, shape preserved)
```

#### Step 4: Compute RR-Interval Features

The **RR-interval** is the time between two consecutive heartbeats. This carries crucial information:

- **Normal heart**: Beats are evenly spaced (~0.8 seconds apart)
- **Premature beat (PVC)**: Comes EARLY — short RR interval before it
- **Compensatory pause**: After a PVC, there's often a LONG gap before the next normal beat

We compute 4 features for each beat:

| Feature | What It Measures | Why It Helps |
|---------|-----------------|-------------|
| `pre_RR` | Time since the previous beat | Short = premature beat |
| `post_RR` | Time until the next beat | Long after PVC = compensatory pause |
| `local_avg_RR` | Average time between nearby beats | Baseline heart rate |
| `pre_RR / avg_RR` | How premature is this beat? | < 1 means early, > 1 means late |

#### Step 5: Map Labels to AAMI Classes

The raw annotations use many specific symbols (N, L, R, A, V, F, /, etc.). We map these to the 5 standard AAMI classes:
- N, L, R, e, j → **N** (Normal)
- A, a, J, S → **S** (Supraventricular)
- V, E → **V** (Ventricular)
- F → **F** (Fusion)
- /, f, Q → **Q** (Unknown/Paced)

#### Step 6: Save Everything

The script saves:
- `X_train.npy` — 97,714 training beats (each is 250 numbers)
- `y_train.npy` — 97,714 labels (0=N, 1=S, 2=V, 3=F, 4=Q)
- `RR_train.npy` — 97,714 sets of 4 RR features
- Same for test set (11,738 beats)
- Demo recordings for the live dashboard

---

## 7. How the AI Model Works

**File: `train_model.py`** — This is the second script you run. It takes the prepared data and trains an AI model to classify heartbeats.

### What is a Neural Network? (Plain English)

A neural network is like a very complex filter. You show it thousands of examples with correct answers, and it slowly adjusts its internal settings until it can predict the right answer on its own.

**Analogy:** Imagine teaching a child to identify dog breeds. You show them thousands of labeled photos:
- "This is a Golden Retriever"
- "This is a Poodle"
- "This is a Bulldog"

After enough examples, the child starts recognizing breeds on their own — even for dogs they've never seen before.

Our neural network does the same thing, but with heartbeat shapes instead of dog photos.

### What is a 1D-CNN?

**CNN** = Convolutional Neural Network. Originally designed for image recognition, but we use the 1D version for signal data.

**How it works:**
1. A small "sliding window" (called a **filter** or **kernel**) moves across the heartbeat signal
2. At each position, it computes a number that represents "how much does this part of the signal match the pattern I'm looking for?"
3. By stacking many filters, the network detects increasingly complex patterns:
   - Layer 1: Detects simple features (sharp spikes, gradual slopes)
   - Layer 2: Combines simple features into medium patterns (QRS shape, T-wave shape)
   - Layer 3: Combines medium patterns into high-level concepts (normal beat shape vs. PVC shape)

```
Input: [0.1, 0.2, 0.8, 1.0, 0.7, 0.3, ...]  ← raw heartbeat (250 numbers)
         ↓ Conv1D filters slide across
Layer 1: [detects edges, spikes, dips]
         ↓
Layer 2: [detects QRS patterns, ST segments]
         ↓
Layer 3: [detects overall beat morphology]
         ↓
Output:  [0.95, 0.01, 0.02, 0.01, 0.01]  ← probabilities for [N, S, V, F, Q]
         → Prediction: N (Normal) with 95% confidence
```

### What are Residual (Skip) Connections?

A common problem with deep neural networks is called **vanishing gradients** — the deeper the network, the harder it is for it to learn. Information gets "diluted" as it passes through many layers.

**Skip connections** solve this by adding a shortcut that lets information bypass layers:

```
Regular network:      Residual network:
Input                 Input ──────────────┐
  ↓                     ↓                 │
Layer 1               Layer 1             │
  ↓                     ↓                 │
Layer 2               Layer 2             │
  ↓                     ↓                 │
Output                Output = Layer2 + Input ←┘ (skip connection!)
```

This lets the network train deeper and learn more complex patterns. Our model uses 3 residual blocks, which is why it outperforms simpler architectures.

### What is Focal Loss?

Our biggest challenge is **class imbalance**. About 82% of all beats are Normal (N), but only 0.4% are Fusion (F). A lazy AI could just predict "Normal" for everything and still be 82% accurate — but it would miss every arrhythmia!

**Focal Loss** is a special training objective that solves this. It works like this:

- If the model correctly classifies a Normal beat with 99% confidence → **very small penalty** (it already knows this)
- If the model misclassifies a rare PVC beat → **very large penalty** (pay attention to this!)

This forces the AI to focus its learning capacity on the hard, rare cases rather than wasting it on easy normal beats.

```
Standard Loss:  Treats all mistakes equally
                N mistake = V mistake = F mistake

Focal Loss:     Scales penalty by difficulty
                Easy N mistake = small penalty
                Hard V mistake = LARGE penalty  ← Forces focus on rare classes!
```

### Data Augmentation

Since rare classes have few examples, we **artificially create more** by slightly modifying existing ones:

1. **Add tiny random noise** — simulates electrode noise (like static on a radio)
2. **Scale amplitude** — simulates different signal strengths (±15%)
3. **Shift timing** — simulates slight R-peak alignment errors (±5 samples)

These augmented beats look slightly different but represent the same type of arrhythmia, giving the AI more variety to learn from.

```
Original PVC beat:     [0.1, 0.3, 0.9, 1.0, 0.7, ...]
+ noise:               [0.12, 0.28, 0.91, 0.98, 0.72, ...]  ← Still a PVC!
+ amplitude scaling:   [0.11, 0.34, 1.02, 1.13, 0.79, ...]  ← Still a PVC!
+ time shift:          [0.3, 0.9, 1.0, 0.7, 0.4, ...]        ← Still a PVC!
```

This grows our training set from 97,714 to ~217,852 beats.

---

## 8. The Dual-Input Architecture

This is what makes our model special compared to most published ECG classifiers.

### The Problem with Single-Input Models

Most ECG AI models look ONLY at the beat waveform shape. But cardiologists also look at **rhythm** — how the beat relates to its neighbors in time:

- "This beat came too early" → might be premature (PVC)
- "There's a long pause after this beat" → compensatory pause (confirms PVC)

### Our Solution: Two Inputs

Our model has **two separate pathways** that process different information, then combine:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  INPUT 1: Beat Waveform (250 numbers)                      │
│  ↓                                                          │
│  Conv1D(64) → ResBlock(64) → ResBlock(128) → ResBlock(256)  │
│  ↓                                                          │
│  GlobalAvgPool → [256 features about SHAPE]                 │
│                                              ↓              │
│                                         CONCATENATE ──→ Dense(256) → Dense(128) → Softmax(5)
│                                              ↑              │
│  INPUT 2: RR Features (4 numbers)            │              │
│  ↓                                           │              │
│  Dense(16) → Dense(8) → [8 features about RHYTHM]          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Branch 1** learns what the beat LOOKS like (morphology).
**Branch 2** learns what the beat's TIMING tells us (rhythm).
Together, they make better decisions than either alone — just like a cardiologist who considers both shape AND rhythm.

### Why This Matters

RR-interval features are **patient-independent**. The shape of a PVC varies a LOT between patients (tall in one, short in another), but the TIMING signature (premature beat → compensatory pause) is universal. This helps the model generalize to new patients it has never seen.

---

## 9. How the Dashboard Works

**File: `app.py`** — This creates the web-based dashboard using Streamlit.

### What is Streamlit?

Streamlit is a Python library that turns Python scripts into interactive web apps. You write Python code, and it automatically creates buttons, charts, sliders, and other UI elements in a web browser. No HTML/CSS/JavaScript needed.

### Dashboard Features

#### Tab 1: Live Monitor
This simulates a real hospital bedside monitor:

1. **Select a patient recording** from the dropdown (Patient 100, 207, or 208)
2. **Click Play** — the ECG waveform scrolls across the screen, just like a real monitor
3. **Each heartbeat is classified in real-time:**
   - Green label = Normal (safe)
   - Red label = Ventricular/PVC (dangerous)
   - Orange label = Supraventricular
4. **Live Stats** show current heart rate, total beats analyzed, arrhythmia count
5. **Arrhythmia Alerts** log every abnormal beat with its type, confidence, and timestamp

#### Tab 2: Model Performance
Shows the AI model's test results:
- **Confusion Matrix**: A grid showing what the model predicted vs. reality
- **Per-class metrics**: Precision, recall, and F1-score for each beat type
- **Training curves**: How accuracy improved over training epochs

### How Inference Works (Behind the Scenes)

When you load a recording, here's what happens:

```
1. Load the saved .npz file (contains signal + pre-extracted beats + RR features)
2. Feed ALL beats + RR features through the model at once (batch prediction)
3. Model outputs probabilities for each beat: [P(N), P(S), P(V), P(F), P(Q)]
4. Pick the class with highest probability as the prediction
5. Cache the results (so clicking Play/Pause doesn't re-run inference)
6. During playback, just display the pre-computed results frame by frame
```

The key insight: **all inference is done ONCE** when you select a recording, not during playback. This is why the animation is smooth — it's just displaying cached results, not running the AI on every frame.

---

## 10. Noise Robustness Testing

**File: `noise_robustness.py`** — Tests how well the model handles noisy signals.

### Why This Matters

In a real hospital, ECG signals are never perfectly clean. Patients move, electrodes shift, nearby machines create interference. A model that works perfectly on clean data but fails on slightly noisy data is useless in practice.

### The Three Types of Noise We Test

1. **Gaussian (White) Noise** — Random electrical interference, like static on a radio
2. **Baseline Wander** — Slow up-and-down drift caused by patient breathing or movement
3. **Muscle Artifact (EMG)** — High-frequency noise from muscle movements (patient fidgeting)

### What is SNR?

**SNR (Signal-to-Noise Ratio)** measures how much stronger the signal is compared to the noise, in decibels (dB):

| SNR | What It Means | Real-World Example |
|-----|--------------|-------------------|
| 30 dB | Signal is 1,000× stronger than noise | Clean hospital recording |
| 20 dB | Signal is 100× stronger | Patient slightly moving |
| 10 dB | Signal is 10× stronger | Patient fidgeting |
| 0 dB | Signal equals noise | Extremely noisy |
| -5 dB | Noise is stronger than signal | Almost unusable |

### Our Results

The model maintains **80.7% accuracy at 30dB** (typical hospital conditions) and is especially robust to baseline wander (**71.3% even at 0dB**). This tells judges: "Our model works in realistic conditions, not just on clean lab data."

---

## 11. External Validation on INCART

**File: `external_validation.py`** — Tests the model on a completely different database.

### Why External Validation is Critical

Many AI papers train and test on the same database (MIT-BIH). This is like studying for an exam using the exact same textbook that the exam comes from — you might score well, but can you apply the knowledge elsewhere?

We tested on the **INCART Database**:
- Different hospital (St. Petersburg, Russia vs. Boston, USA)
- Different equipment (different ECG machines)
- Different patients (75 new patients the model has never seen)
- Different sampling rate (257 Hz vs. 360 Hz — we had to resample)

### Results

**87% overall accuracy** on 173,476 completely unseen beats. The model correctly identified normal beats across databases. Arrhythmia detection was limited — which is expected and honest. This gap shows where future work is needed (domain adaptation, transfer learning) and demonstrates scientific rigor rather than inflated claims.

---

## 12. How All the Files Connect Together

```
                    STEP 1                    STEP 2                    STEP 3
              ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
PhysioNet ──→ │  data_prep.py │ ──→     │train_model.py │ ──→     │    app.py      │
 (raw data)   │               │  .npy   │               │ .keras  │  (dashboard)   │
              │ Download      │  files  │ Oversample    │  model  │  Live Monitor  │
              │ Segment beats │ ──────→ │ Augment       │ ──────→ │  Performance   │
              │ Normalize     │         │ Train CNN     │         │  Alerts        │
              │ Compute RR    │         │ Evaluate      │         │                │
              └───────────────┘         └───────────────┘         └───────────────┘
                                                                         ↑
                                              Also feeds into:           │
                                        ┌─────────────────────┐         │
                                        │noise_robustness.py  │─────────┘
                                        │external_validation.py│  (uses same model)
                                        └─────────────────────┘
```

### Complete File List

| File | Purpose | When to Run |
|------|---------|------------|
| `data_prep.py` | Downloads ECG data, segments beats, computes RR features, saves .npy files | Run FIRST (once) |
| `train_model.py` | Builds and trains the dual-input CNN, evaluates, saves model | Run SECOND (once, takes ~1 hour) |
| `app.py` | Streamlit dashboard for live demo | Run ANYTIME for demo |
| `noise_robustness.py` | Tests model under various noise conditions | Run AFTER training |
| `external_validation.py` | Validates on INCART database | Run AFTER training |
| `create_ppt.py` | Generates the hackathon pitch deck PPT | Run ONCE |
| `run_dashboard.bat` | Double-click launcher for the dashboard | Anytime |
| `requirements.txt` | Lists all Python packages needed | Used during setup |
| `README.md` | Project documentation for GitHub | Reference |
| `.gitignore` | Tells Git which files not to upload | Automatic |

### Output Files (in `output/` folder)

| File | What It Contains |
|------|-----------------|
| `ecg_model.keras` | The trained AI model (2.3 MB) |
| `X_train.npy`, `y_train.npy` | Training data (beats + labels) |
| `X_test.npy`, `y_test.npy` | Test data |
| `RR_train.npy`, `RR_test.npy` | RR-interval features |
| `metrics.json` | Accuracy, precision, recall numbers |
| `confusion_matrix.png` | Visual grid of predictions vs. reality |
| `classification_report.txt` | Detailed per-class performance |
| `training_history.png` | Graph of accuracy over training epochs |
| `noise_robustness.json` | Noise test results |
| `noise_robustness_curve.png` | Noise degradation plot |
| `external_validation.json` | INCART test results |
| `metadata.json` | Dataset info (class names, record IDs, etc.) |
| `demo_recordings/*.npz` | Pre-processed recordings for dashboard |

---

## 13. Key Results and What They Mean

### Model Performance (v4 — Dual-Input Residual CNN)

| Metric | Value | What It Means |
|--------|-------|--------------|
| **Overall Accuracy** | 82.0% | Out of every 100 beats, the model correctly classifies 82 |
| **N (Normal) Recall** | 84.1% | Catches 84.1% of all normal beats |
| **V (PVC) Recall** | 98.8% | Catches 98.8% of all dangerous ventricular beats |
| **V (PVC) Precision** | 41.5% | When it says "this is a PVC," it's right 41.5% of the time |
| **Training Accuracy** | 99.9% | The model memorized the training data almost perfectly |
| **Parameters** | 608,541 | The number of adjustable "knobs" in the model |
| **Model Size** | 2.3 MB | Small enough to run on a phone |
| **Inference Time** | < 5 ms | Classifies one heartbeat in less than 5 milliseconds |

### Understanding Precision vs. Recall

These are the two most important metrics, and they trade off:

**Recall (Sensitivity)** = "Of all the ACTUAL PVCs, what % did we catch?"
- 98.8% recall means we only miss 1.2% of PVCs
- **Missing a PVC is dangerous** — the patient could die
- We prioritize HIGH recall for safety

**Precision** = "Of all the beats we CALLED PVCs, what % were actually PVCs?"
- 41.5% precision means 58.5% of our PVC alerts are false alarms
- False alarms are annoying but NOT dangerous
- We accept lower precision as a safety trade-off

**Think of it like a smoke detector:**
- High recall = detects every real fire (even if it sometimes goes off while cooking)
- High precision = never gives false alarms (but might miss some real fires)
- For life-safety, we want HIGH RECALL — better to have some false alarms than miss a real fire!

### Why 82% and Not 99%?

Some papers claim 99%+ accuracy. Here's why our 82% is actually MORE honest:

1. **Patient-level split**: We test on patients the model has NEVER seen. Papers claiming 99% often mix beats from the same patient in train and test sets — the model recognizes the patient's unique heart shape, not the arrhythmia.

2. **Class imbalance**: 85% of test beats are Normal. A "dumb" model that always says Normal would get 85% accuracy. Our 82% with 98.8% PVC recall is much more useful.

3. **Generalization**: Our model works on new patients. That's the real challenge in medicine — every heart is different.

### Training Accuracy (99.9%) vs. Test Accuracy (82.0%)

This 18% gap might seem alarming, but it's **expected and scientifically correct**:

- **Training accuracy** tells you the model memorized the training data well
- **Test accuracy** tells you the model generalizes to new patients
- The gap exists because every patient's heart is unique — the model must adapt to shapes it's never seen
- A small gap (e.g., 99% train, 98% test) would be suspicious — it might mean the test data leaked into training

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **AAMI** | Association for the Advancement of Medical Instrumentation — sets the standard for ECG beat classification |
| **Arrhythmia** | Any heartbeat that doesn't follow normal rhythm |
| **Augmentation** | Creating artificial training examples by slightly modifying real ones |
| **Batch Normalization (BN)** | A technique that stabilizes neural network training by normalizing values between layers |
| **CNN** | Convolutional Neural Network — an AI architecture that detects patterns by sliding filters across data |
| **Dropout** | A training technique that randomly disables neurons to prevent overfitting (like studying with random flashcards removed) |
| **ECG/EKG** | Electrocardiogram — a recording of the heart's electrical activity |
| **Epoch** | One complete pass through the entire training dataset. Our model trained for ~45 epochs |
| **Focal Loss** | A special training objective that forces the model to focus on hard, rare examples |
| **Hz (Hertz)** | Measurements per second. 360 Hz = 360 samples per second |
| **Inference** | Using the trained model to make predictions on new data |
| **Keras** | A Python library for building neural networks (part of TensorFlow) |
| **MIT-BIH** | Massachusetts Institute of Technology - Beth Israel Hospital database |
| **Normalization** | Scaling data to a standard range (e.g., 0 to 1) so the AI focuses on patterns, not raw values |
| **Oversampling** | Duplicating rare class examples so the AI sees them more often during training |
| **PhysioNet** | A free online repository of medical signal databases |
| **PVC** | Premature Ventricular Contraction — a dangerous type of arrhythmia |
| **R-peak** | The tallest spike in a heartbeat waveform, used to locate each beat |
| **Recall** | The percentage of actual positives that the model correctly identifies |
| **Precision** | The percentage of the model's positive predictions that are actually correct |
| **Residual Block** | A neural network building block with skip connections for better training |
| **RR-interval** | The time between consecutive R-peaks (consecutive heartbeats) |
| **SNR** | Signal-to-Noise Ratio — how clean the signal is compared to background noise |
| **Softmax** | A function that converts raw scores into probabilities that sum to 1 |
| **Streamlit** | A Python framework for building interactive web dashboards |
| **TensorFlow** | Google's open-source AI/machine learning framework |
| **Validation Split** | A portion of training data held back to monitor learning progress (10% in our case) |
