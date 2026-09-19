#!/usr/bin/env python3
"""
ECG Arrhythmia Detector - Streamlit Dashboard
=============================================
Real-time ECG playback simulation with AI-powered arrhythmia detection.
Replays pre-recorded hospital ECG signals while a trained 1D-CNN classifies
each heartbeat as normal or abnormal, highlighting flagged beats on a
scrolling waveform chart and displaying alerts with confidence scores.

Run with:
    streamlit run app.py

⚠ RESEARCH PROTOTYPE - Not a validated medical device.
   For demonstration and educational purposes only.
"""

import os
import json
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf


# ── Configuration ─────────────────────────────────────────────────────

OUTPUT_DIR = "output"
DEMO_DIR = os.path.join(OUTPUT_DIR, "demo_recordings")
MODEL_PATH = os.path.join(OUTPUT_DIR, "ecg_model.keras")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
CM_IMAGE_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
HISTORY_IMAGE_PATH = os.path.join(OUTPUT_DIR, "training_history.png")

CLASS_NAMES = ["N", "S", "V", "F", "Q"]

# Human-readable names for each AAMI class
CLASS_DESCRIPTIONS = {
    "N": "Normal Beat",
    "S": "Supraventricular Ectopic",
    "V": "Ventricular Ectopic (PVC)",
    "F": "Fusion Beat",
    "Q": "Unknown / Paced",
}

# Clinical significance descriptions for alert context
CLASS_CLINICAL = {
    "N": "Normal sinus rhythm - no action needed.",
    "S": "Beat originating above the ventricles. Frequent occurrences may indicate atrial fibrillation risk.",
    "V": "Premature ventricular contraction. Frequent PVCs can indicate ventricular tachycardia risk.",
    "F": "Hybrid beat from simultaneous normal and ventricular activation.",
    "Q": "Unclassifiable or pacemaker-generated beat.",
}

# Color scheme for beat classes (medical-convention inspired)
CLASS_COLORS = {
    "N": "#2ecc71",   # green  - all clear
    "S": "#f39c12",   # orange - attention
    "V": "#e74c3c",   # red    - danger
    "F": "#9b59b6",   # purple - unusual
    "Q": "#95a5a6",   # gray   - unknown
}

# Beat class icons for visual emphasis
CLASS_ICONS = {
    "N": "✅",
    "S": "⚠️",
    "V": "🔴",
    "F": "🟣",
    "Q": "❓",
}

FS = 360        # sampling frequency (Hz)
PRE_PEAK = 90   # samples before R-peak in beat window
POST_PEAK = 160  # samples after R-peak in beat window

# Animation parameters
CHUNK_SIZE = 90          # samples to advance per animation frame (~0.25s of signal)
VISIBLE_WINDOW = 3600    # samples visible at once (~10 seconds of ECG)
ANIMATION_DELAY = 0.05   # seconds between animation frames (~20 FPS)


# ── Data & Model Loading (Cached) ────────────────────────────────────

@st.cache_resource
def load_model():
    """
    Load the trained 1D-CNN model from disk.

    @st.cache_resource ensures the model is loaded only ONCE and shared
    across all Streamlit reruns/sessions, keeping inference fast.
    """
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_metrics():
    """Load pre-computed evaluation metrics from the training step."""
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_demo_recording(record_id):
    """
    Load a pre-saved demo recording for live playback.

    Returns a dict with:
        signal: full continuous ECG signal (raw amplitude)
        peaks: R-peak sample indices
        labels: true AAMI class labels (for comparison)
        beat_segments: pre-extracted & normalized 250-sample beat windows
    """
    path = os.path.join(DEMO_DIR, f"record_{record_id}.npz")
    data = np.load(path)
    return {
        "signal": data["signal"],
        "peaks": data["peaks"],
        "labels": data["labels"],
        "beat_segments": data["beat_segments"],
        "rr_features": data["rr_features"] if "rr_features" in data else np.zeros((len(data["peaks"]), 4), dtype=np.float32),
    }


@st.cache_data
def predict_all_beats(_model, beat_segments, rr_features, record_id):
    # Run model inference on ALL beats in a recording at once.
    # We pre-compute predictions for the entire recording up front so the
    # playback animation is silky smooth - no inference delay per frame.
    # The _model parameter has an underscore prefix to tell Streamlit
    # not to try hashing it (models are not hashable).
    X = beat_segments[..., np.newaxis]
    RR = rr_features.astype(np.float32)
    probs = _model.predict([X, RR], verbose=0)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    return preds, confidences


def get_available_demos():
    """Scan the demo directory for available patient recordings."""
    demos = {}
    if os.path.exists(DEMO_DIR):
        for f in sorted(os.listdir(DEMO_DIR)):
            if f.startswith("record_") and f.endswith(".npz"):
                rec_id = f.replace("record_", "").replace(".npz", "")
                demos[rec_id] = f"Patient {rec_id}"
    return demos


# ── ECG Visualization ────────────────────────────────────────────────

def create_ecg_plot(signal, current_pos, visible_window, peaks, predictions,
                    confidences, title="ECG Signal"):
    """
    Create a Plotly figure showing the scrolling ECG waveform.

    The chart displays a 10-second window of ECG signal that scrolls
    left-to-right as the animation progresses. Abnormal beats are
    highlighted with colored shaded regions and annotation labels.

    This mimics a real hospital bedside monitor display.
    """
    # Calculate visible range
    start = max(0, current_pos - visible_window)
    end = current_pos

    # Time axis in seconds (more intuitive than sample numbers)
    time_axis = np.arange(start, end) / FS
    visible_signal = signal[start:end]

    fig = go.Figure()

    # ── Main ECG trace ──
    # Cyan color on dark background mimics clinical ECG monitors
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=visible_signal,
        mode="lines",
        line=dict(color="#00d4ff", width=1.5),
        name="ECG Signal",
        hoverinfo="skip",
    ))

    # ── Highlight abnormal beats ──
    # For each beat within the visible window, if it's predicted as
    # abnormal, draw a shaded region and add an annotation arrow
    for i, peak in enumerate(peaks):
        if start <= peak < end and i < len(predictions):
            pred_class = CLASS_NAMES[predictions[i]]

            if pred_class != "N":
                color = CLASS_COLORS[pred_class]
                conf = confidences[i] * 100

                # Shaded region over the beat's duration
                beat_start = max(start, peak - PRE_PEAK)
                beat_end = min(end, peak + POST_PEAK)
                fig.add_vrect(
                    x0=beat_start / FS, x1=beat_end / FS,
                    fillcolor=color, opacity=0.15,
                    layer="below", line_width=0,
                )

                # Annotation marker pointing to the R-peak
                fig.add_annotation(
                    x=peak / FS,
                    y=float(signal[peak]),
                    text=f"{CLASS_ICONS.get(pred_class, '⚠')} {pred_class} ({conf:.0f}%)",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor=color,
                    font=dict(color=color, size=11, family="Arial Black"),
                    bgcolor="rgba(10, 10, 10, 0.85)",
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=3,
                    ay=-40,
                )

    # ── Layout styling (dark clinical monitor theme) ──
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#ffffff")),
        xaxis=dict(
            title="Time (seconds)",
            showgrid=True,
            gridcolor="rgba(50, 50, 50, 0.5)",
            gridwidth=1,
            range=[start / FS, end / FS],
            color="#aaaaaa",
        ),
        yaxis=dict(
            title="Amplitude (mV)",
            showgrid=True,
            gridcolor="rgba(50, 50, 50, 0.5)",
            gridwidth=1,
            color="#aaaaaa",
        ),
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        font=dict(color="#ffffff"),
        height=420,
        margin=dict(l=60, r=20, t=50, b=50),
        showlegend=False,
    )

    return fig


def create_beat_comparison_plot(abnormal_beat, normal_beat, abn_class):
    """
    Create a side-by-side comparison of an abnormal beat vs. a normal beat.

    This is the "explainability" feature - by showing both waveforms
    together, even non-cardiologists can visually see differences like:
    - PVCs have wider, taller QRS complexes
    - Supraventricular beats have abnormal P-waves
    - Fusion beats have hybrid morphology
    """
    fig = go.Figure()

    x = np.arange(len(normal_beat))

    # Normal beat (green, reference)
    fig.add_trace(go.Scatter(
        x=x, y=normal_beat,
        mode="lines",
        line=dict(color="#2ecc71", width=2, dash="dot"),
        name="Normal Beat (reference)",
        opacity=0.7,
    ))

    # Abnormal beat (colored by class)
    fig.add_trace(go.Scatter(
        x=x, y=abnormal_beat,
        mode="lines",
        line=dict(color=CLASS_COLORS.get(abn_class, "#e74c3c"), width=2.5),
        name=f"{CLASS_DESCRIPTIONS.get(abn_class, abn_class)} (flagged)",
    ))

    fig.update_layout(
        title=dict(
            text=f"🔍 Beat Comparison: {CLASS_DESCRIPTIONS.get(abn_class, abn_class)} vs. Normal",
            font=dict(size=14, color="#ffffff"),
        ),
        xaxis=dict(title="Sample Index", showgrid=True, gridcolor="#333", color="#aaa"),
        yaxis=dict(title="Normalized Amplitude", showgrid=True, gridcolor="#333", color="#aaa"),
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        font=dict(color="#ffffff"),
        height=280,
        margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=10),
        ),
    )

    return fig


# ── Dashboard Layout ─────────────────────────────────────────────────

def main():
    # ── Page configuration ──
    st.set_page_config(
        page_title="ECG Arrhythmia Detector",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Custom CSS for medical-dashboard styling ──
    st.markdown("""
    <style>
    /* Dark theme overrides for clinical look */
    .stMetric { background-color: #111; border-radius: 8px; padding: 10px; }
    .stMetric label { color: #888 !important; }
    .alert-card {
        padding: 10px 14px;
        margin: 4px 0;
        border-radius: 6px;
        background-color: rgba(20, 20, 20, 0.9);
        border-left: 4px solid;
    }
    .disclaimer {
        background-color: #1a1a2e;
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.85em;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    # ❤️ Real-Time ECG Arrhythmia Detector
    **AI-powered heartbeat classification** using a 1D Convolutional Neural Network trained
    on the MIT-BIH Arrhythmia Database - real hospital ECG recordings annotated beat-by-beat
    by board-certified cardiologists.
    """)

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Research Prototype</strong> - This is a hackathon demonstration project, NOT a validated
    medical device. Not intended for clinical diagnosis. See the Model Performance tab for validation details.
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ──
    try:
        model = load_model()
    except Exception as e:
        st.error(
            f"❌ **Could not load model.** Make sure you've run the pipeline first:\n\n"
            f"```bash\npip install -r requirements.txt\npython data_prep.py\npython train_model.py\n```\n\n"
            f"Error: `{e}`"
        )
        return

    # ── Check for demo recordings ──
    demos = get_available_demos()
    if not demos:
        st.error(
            "❌ **No demo recordings found.** Run `python data_prep.py` first to prepare patient data."
        )
        return

    # ── Tab navigation ──
    tab_live, tab_performance = st.tabs([
        "🫀 Live Monitor",
        "📊 Model Performance",
    ])

    # ══════════════════════════════════════════════════════════════════
    #  TAB 1: LIVE ECG MONITOR
    # ══════════════════════════════════════════════════════════════════
    with tab_live:
        # ── Patient selector & controls row ──
        col_select, col_controls = st.columns([2, 3])

        with col_select:
            selected_id = st.selectbox(
                "🏥 Select Patient Recording",
                options=list(demos.keys()),
                format_func=lambda x: demos[x],
                help="Choose a pre-loaded test-set recording for live playback. "
                     "Patient 100 is mostly normal; Patients 207/208 have arrhythmias.",
            )

        # ── Load recording & pre-compute predictions ──
        recording = load_demo_recording(selected_id)
        signal = recording["signal"]
        peaks = recording["peaks"]
        true_labels = recording["labels"]
        beat_segments = recording["beat_segments"]
        rr_features = recording["rr_features"]

        # Pre-compute all predictions for smooth playback
        # (this runs ONCE per recording selection, then is cached)
        preds, confs = predict_all_beats(model, beat_segments, rr_features, selected_id)

        # ── Initialize session state for animation ──
        if "position" not in st.session_state:
            st.session_state.position = VISIBLE_WINDOW
        if "playing" not in st.session_state:
            st.session_state.playing = False
        if "selected_record" not in st.session_state:
            st.session_state.selected_record = selected_id

        # Reset animation when patient changes
        if st.session_state.selected_record != selected_id:
            st.session_state.position = VISIBLE_WINDOW
            st.session_state.playing = False
            st.session_state.selected_record = selected_id

        # ── Playback controls ──
        with col_controls:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("▶️ Play", use_container_width=True, type="primary"):
                    st.session_state.playing = True
            with c2:
                if st.button("⏸ Pause", use_container_width=True):
                    st.session_state.playing = False
            with c3:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.position = VISIBLE_WINDOW
                    st.session_state.playing = False
            with c4:
                speed = st.select_slider(
                    "Speed",
                    options=["0.5x", "1x", "2x", "4x", "8x"],
                    value="2x",
                    help="Animation playback speed multiplier",
                )
                speed_mult = {"0.5x": 0.5, "1x": 1, "2x": 2, "4x": 4, "8x": 8}[speed]

        # ── Current position ──
        pos = st.session_state.position

        # ── Identify beats visible so far ──
        visible_mask = peaks <= pos
        num_visible = int(np.sum(visible_mask))
        visible_preds = preds[:num_visible]
        visible_confs = confs[:num_visible]
        visible_peaks = peaks[:num_visible]
        visible_true = true_labels[:num_visible]

        total_beats = len(visible_preds)
        normal_count = int(np.sum(visible_preds == 0))
        abnormal_count = total_beats - normal_count

        # ── Heart rate calculation ──
        # Estimated from the average R-R interval of recent beats
        if len(visible_peaks) >= 2:
            recent_peaks = visible_peaks[-min(10, len(visible_peaks)):]
            rr_intervals = np.diff(recent_peaks) / FS  # seconds between beats
            avg_rr = np.mean(rr_intervals)
            heart_rate = int(60 / avg_rr) if avg_rr > 0 else 0
        else:
            heart_rate = 0

        # ── Progress bar ──
        progress = min(pos / len(signal), 1.0)
        st.progress(progress, text=f"📍 Recording progress: {progress*100:.0f}% "
                    f"({pos/FS:.1f}s / {len(signal)/FS:.1f}s)")

        # ── ECG Chart ──
        chart_placeholder = st.empty()
        fig = create_ecg_plot(
            signal, pos, VISIBLE_WINDOW, peaks, preds, confs,
            title=f"Patient {selected_id} - ECG Monitor",
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"ecg_{pos}")

        # ── Stats & Alerts side-by-side ──
        col_stats, col_alerts = st.columns([1, 2])

        # ── Live Statistics Panel ──
        with col_stats:
            st.markdown("### 📈 Live Stats")

            # Heart rate with emoji indicator
            hr_color = "normal" if 60 <= heart_rate <= 100 else "off"
            st.metric(
                "💓 Heart Rate",
                f"{heart_rate} BPM" if heart_rate > 0 else "-",
                help="Estimated from average R-R interval of last 10 beats",
            )

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Total Beats", f"{total_beats}")
            with m2:
                st.metric("⏱ Time", f"{pos/FS:.1f}s")

            m3, m4 = st.columns(2)
            with m3:
                st.metric("✅ Normal", f"{normal_count}")
            with m4:
                st.metric(
                    "⚠️ Abnormal", f"{abnormal_count}",
                    delta=f"+{abnormal_count}" if abnormal_count > 0 else None,
                    delta_color="inverse",
                )

            # Abnormal rate
            if total_beats > 0:
                abn_rate = abnormal_count / total_beats * 100
                st.metric("Abnormal Rate", f"{abn_rate:.1f}%")

        # ── Alert Panel ──
        with col_alerts:
            st.markdown("### 🚨 Arrhythmia Alerts")

            # Find abnormal beats in visible range
            abnormal_indices = np.where(visible_preds != 0)[0]

            if len(abnormal_indices) == 0:
                st.info("✅ No abnormal beats detected yet. All heartbeats appear normal.")
            else:
                st.caption(f"Showing last {min(15, len(abnormal_indices))} of "
                           f"{len(abnormal_indices)} total alerts")

                # Show most recent alerts first (reverse chronological)
                for idx in reversed(abnormal_indices[-15:]):
                    pred_cls = CLASS_NAMES[visible_preds[idx]]
                    conf_pct = visible_confs[idx] * 100
                    peak_time = visible_peaks[idx] / FS
                    color = CLASS_COLORS[pred_cls]
                    icon = CLASS_ICONS[pred_cls]
                    true_cls = CLASS_NAMES[visible_true[idx]]

                    st.markdown(
                        f'<div class="alert-card" style="border-left-color: {color};">'
                        f'<strong style="color:{color}">{icon} {CLASS_DESCRIPTIONS[pred_cls]}</strong>'
                        f' <span style="color:#888;">({pred_cls})</span><br>'
                        f'<span style="color:#ccc;">Confidence: <strong>{conf_pct:.1f}%</strong>'
                        f' &nbsp;|&nbsp; Time: <strong>{peak_time:.2f}s</strong>'
                        f' &nbsp;|&nbsp; True: {true_cls}</span></div>',
                        unsafe_allow_html=True,
                    )

        # ── Explainability: Beat Comparison ──
        if len(abnormal_indices) > 0:
            st.markdown("---")
            st.markdown("### 🔍 Why Was This Beat Flagged?")
            st.caption(
                "Comparing the most recently flagged abnormal beat (colored) to a "
                "typical normal beat (green, dashed) from the same patient. "
                "Notice differences in QRS width, amplitude, and overall morphology."
            )

            # Get the last abnormal beat
            last_abn_idx = abnormal_indices[-1]
            abn_class = CLASS_NAMES[preds[last_abn_idx]]
            abnormal_beat = beat_segments[last_abn_idx]

            # Find a normal beat for comparison (pick one from the middle of the recording)
            normal_indices = np.where(preds == 0)[0]
            if len(normal_indices) > 0:
                # Use median normal beat for a "typical" reference
                normal_idx = normal_indices[len(normal_indices) // 2]
                normal_beat = beat_segments[normal_idx]

                # Side-by-side comparison
                fig_compare = create_beat_comparison_plot(
                    abnormal_beat, normal_beat, abn_class
                )
                st.plotly_chart(fig_compare, use_container_width=True)

                # Clinical context
                st.info(
                    f"**{CLASS_DESCRIPTIONS[abn_class]}:** "
                    f"{CLASS_CLINICAL.get(abn_class, 'No additional info.')}"
                )

        # ── Animation loop ──
        # If playing and not at end of recording, advance position and rerun
        if st.session_state.playing and pos < len(signal):
            advance = int(CHUNK_SIZE * speed_mult)
            st.session_state.position = min(pos + advance, len(signal))
            time.sleep(ANIMATION_DELAY)
            st.rerun()
        elif pos >= len(signal):
            st.session_state.playing = False
            st.success("✅ Playback complete! All beats in this recording have been analyzed.")

    # ══════════════════════════════════════════════════════════════════
    #  TAB 2: MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════
    with tab_performance:
        st.markdown("### 📊 Model Performance on Held-Out Test Set")
        st.caption(
            "These metrics were computed on patient recordings the model **never saw** "
            "during training. The train/test split is done at the patient level to "
            "prevent data leakage."
        )

        try:
            metrics = load_metrics()
        except Exception:
            st.error("❌ Metrics not found. Run `python train_model.py` first.")
            return

        # ── Key metrics summary ──
        col_acc, col_train, col_test, col_epochs = st.columns(4)
        with col_acc:
            st.metric("🎯 Overall Accuracy", f"{metrics['accuracy']*100:.1f}%")
        with col_train:
            st.metric("📚 Training Samples", f"{metrics['train_size']:,}")
        with col_test:
            st.metric("🧪 Test Samples", f"{metrics['test_size']:,}")
        with col_epochs:
            st.metric("🔄 Epochs Trained", f"{metrics.get('epochs_trained', '-')}")

        st.markdown("---")

        # ── Clinical importance explanation ──
        st.markdown("""
        #### Understanding the Metrics

        | Metric | What It Means | Why It Matters Clinically |
        |--------|--------------|--------------------------|
        | **Recall** (Sensitivity) | Of all real arrhythmias, what % did we catch? | **Most critical** - a missed arrhythmia could be fatal |
        | **Precision** | Of all our alarms, what % were real? | High false alarms cause "alarm fatigue" in ICUs |
        | **F1-Score** | Harmonic mean of precision & recall | Balanced measure when both matter |
        | **Support** | Number of test samples per class | Shows class imbalance in the data |
        """)

        # ── Per-class metrics table ──
        st.markdown("#### Per-Class Performance")

        per_class = metrics.get("per_class", {})
        if per_class:
            table_data = []
            for cls in CLASS_NAMES:
                if cls in per_class:
                    m = per_class[cls]
                    table_data.append({
                        "Class": f"{CLASS_ICONS.get(cls, '')} {cls} - {CLASS_DESCRIPTIONS[cls]}",
                        "Precision": f"{m['precision']*100:.1f}%",
                        "Recall": f"{m['recall']*100:.1f}%",
                        "F1-Score": f"{m['f1_score']*100:.1f}%",
                        "Support": f"{m['support']:,}",
                    })

            st.table(table_data)

        st.markdown("---")

        # ── Confusion Matrix ──
        st.markdown("#### Confusion Matrix")
        st.caption("Rows = true labels, columns = predicted labels. "
                   "Diagonal = correct predictions. Off-diagonal = errors.")

        col_cm, col_cm_info = st.columns([2, 1])

        with col_cm:
            if os.path.exists(CM_IMAGE_PATH):
                st.image(CM_IMAGE_PATH, use_container_width=True)
            else:
                st.warning("Confusion matrix image not found.")

        with col_cm_info:
            st.markdown("""
            **How to read this:**
            - Dark blue diagonal cells = correct predictions (good!)
            - Off-diagonal cells = misclassifications
            - Each row sums to 100% of that true class

            **Key things to check:**
            - Is V (PVC) recall high? Missing PVCs is dangerous.
            - Are S beats getting confused with N? Common challenge.
            - Is the model biased toward predicting N? (Class imbalance issue)
            """)

        # ── Training History ──
        if os.path.exists(HISTORY_IMAGE_PATH):
            st.markdown("---")
            st.markdown("#### Training History")
            st.caption("Accuracy and loss curves over training epochs. "
                       "Validation curves should track training curves without diverging (overfitting).")
            st.image(HISTORY_IMAGE_PATH, use_container_width=True)

        # ── Limitations ──
        st.markdown("---")
        st.markdown("#### ⚠️ Important Limitations")
        st.markdown("""
        This is a **research prototype / hackathon demo**, NOT a validated medical device:

        1. **Retrospective data only** - Validated on the MIT-BIH Arrhythmia Database
           (recorded 1975–1979 at Beth Israel Hospital). Has not been tested on
           prospective real-time data from modern ECG devices.

        2. **Limited patient demographics** - MIT-BIH contains 48 patients from a single
           hospital. Real-world deployment would require validation across diverse
           populations, ages, and comorbidities.

        3. **Simplified classification** - Uses 5 AAMI classes. Real clinical ECG
           interpretation involves dozens of rhythm and morphology patterns, 12-lead
           analysis, and clinical context (symptoms, history, medications).

        4. **No regulatory clearance** - Would require FDA 510(k) or De Novo
           classification (or equivalent EU MDR / CE marking) before any clinical use.

        5. **Single-lead analysis** - Uses only Lead MLII. Clinical ECG interpretation
           uses 12 leads for comprehensive cardiac assessment.

        6. **Patient-level splitting caveat** - While we split by patient, all patients
           come from the same hospital/era, so there may be site-specific biases
           that wouldn't generalize to other clinical settings.
        """)

        st.markdown("""
        ---
        *Built with TensorFlow, Streamlit, and the MIT-BIH Arrhythmia Database from PhysioNet.*
        *Model architecture: 1D-CNN with 3 convolutional layers and batch normalization.*
        """)


if __name__ == "__main__":
    main()
