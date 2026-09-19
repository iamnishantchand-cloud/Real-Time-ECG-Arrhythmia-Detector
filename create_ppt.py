"""
Generate the Synapse Hackathon Pitch Deck for ECG Arrhythmia Detector.
Follows the exact 8-slide format from Synapse_ppt_format.pdf.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---------- colour palette ----------
BG_DARK   = RGBColor(0x0A, 0x0A, 0x1A)   # deep navy
BG_CARD   = RGBColor(0x12, 0x12, 0x2A)   # card background
ACCENT    = RGBColor(0x00, 0xD4, 0xAA)   # teal accent
ACCENT2   = RGBColor(0xE7, 0x4C, 0x3C)   # red accent
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT     = RGBColor(0xCC, 0xCC, 0xCC)
MUTED     = RGBColor(0x88, 0x88, 0x99)
GREEN     = RGBColor(0x2E, 0xCC, 0x71)
ORANGE    = RGBColor(0xF3, 0x9C, 0x12)

OUTPUT_DIR = r"C:\Users\DELL\ecg-arrhythmia-detector"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ECG_Arrhythmia_Detector_Pitch.pptx")

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SLIDE_W = prs.slide_width
SLIDE_H = prs.slide_height


# ========== helper functions ==========

def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape_rect(slide, left, top, width, height, fill_color, border_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    # Round corners
    shape.adjustments[0] = 0.05
    return shape


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_bullet_frame(slide, left, top, width, height, items, font_size=16,
                     color=LIGHT, bullet_color=ACCENT, title=None, title_size=22):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    if title:
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(title_size)
        p.font.color.rgb = ACCENT
        p.font.bold = True
        p.font.name = "Calibri"
        p.space_after = Pt(12)
        start_new = True
    else:
        start_new = False

    for i, item in enumerate(items):
        if start_new or i > 0:
            p = tf.add_paragraph()
        else:
            p = tf.paragraphs[0]
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = Pt(8)
        p.level = 0
    return txBox


def add_accent_line(slide, left, top, width):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Pt(4))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()
    return shape


def add_metric_card(slide, left, top, width, height, value, label, value_color=ACCENT):
    card = add_shape_rect(slide, left, top, width, height, BG_CARD, border_color=RGBColor(0x33, 0x33, 0x55))
    # Value
    add_text_box(slide, left + Inches(0.15), top + Inches(0.15), width - Inches(0.3), Inches(0.6),
                 value, font_size=28, color=value_color, bold=True, alignment=PP_ALIGN.CENTER)
    # Label
    add_text_box(slide, left + Inches(0.15), top + Inches(0.7), width - Inches(0.3), Inches(0.5),
                 label, font_size=12, color=MUTED, alignment=PP_ALIGN.CENTER)
    return card


# =====================================================================
# SLIDE 1: TITLE / PITCH DECK
# =====================================================================
slide1 = prs.slides.add_slide(prs.slide_layouts[6])  # blank
set_slide_bg(slide1, BG_DARK)

# Accent line at top
add_accent_line(slide1, Inches(0), Inches(0), SLIDE_W)

# ECG heartbeat icon (unicode heart)
add_text_box(slide1, Inches(4.5), Inches(1.2), Inches(4.3), Inches(1.2),
             "\u2764", font_size=60, color=ACCENT2, alignment=PP_ALIGN.CENTER)

# Title
add_text_box(slide1, Inches(1.5), Inches(2.2), Inches(10.3), Inches(1.2),
             "Real-Time AI ECG Arrhythmia Detector",
             font_size=44, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

# Subtitle
add_text_box(slide1, Inches(2), Inches(3.4), Inches(9.3), Inches(0.8),
             "Deep Learning-Powered Cardiac Monitoring for Life-Saving Arrhythmia Detection",
             font_size=22, color=LIGHT, alignment=PP_ALIGN.CENTER)

# Accent line
add_accent_line(slide1, Inches(5.5), Inches(4.4), Inches(2.3))

# Track & Team
add_text_box(slide1, Inches(3), Inches(4.8), Inches(7.3), Inches(0.5),
             "Track: Healthcare / AI Innovation",
             font_size=20, color=ACCENT, alignment=PP_ALIGN.CENTER)

add_text_box(slide1, Inches(3), Inches(5.4), Inches(7.3), Inches(0.5),
             "Team: [Your Team Name]",
             font_size=20, color=MUTED, alignment=PP_ALIGN.CENTER)

# Bottom bar
add_text_box(slide1, Inches(3), Inches(6.5), Inches(7.3), Inches(0.4),
             "SYNAPSE HACKATHON 2026  |  PITCH DECK",
             font_size=14, color=MUTED, alignment=PP_ALIGN.CENTER)


# =====================================================================
# SLIDE 2: PROBLEM STATEMENT & EXISTING GAP
# =====================================================================
slide2 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide2, BG_DARK)
add_accent_line(slide2, Inches(0), Inches(0), SLIDE_W)

add_text_box(slide2, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "PROBLEM STATEMENT & EXISTING GAP",
             font_size=32, color=WHITE, bold=True)
add_accent_line(slide2, Inches(0.8), Inches(1.1), Inches(3))

# Left column - The Problem
add_shape_rect(slide2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3), BG_CARD,
               border_color=RGBColor(0x33, 0x33, 0x55))

add_bullet_frame(slide2, Inches(1.1), Inches(1.7), Inches(5), Inches(5),
                 [
                     "Cardiovascular diseases are the #1 cause of death globally, killing 17.9 million people annually (WHO).",
                     "Arrhythmias are irregular heartbeats that can cause sudden cardiac death if not detected early.",
                     "In ICUs, nurses monitor ECG screens 24/7 - but 'alarm fatigue' causes up to 90% of alerts to be ignored.",
                     "A single missed Ventricular ectopic beat (PVC) in a critical patient can be fatal.",
                     "Rural and under-resourced hospitals often lack trained cardiologists for continuous ECG interpretation."
                 ],
                 title="The Problem", font_size=15, color=LIGHT)

# Right column - Existing Gaps
add_shape_rect(slide2, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), BG_CARD,
               border_color=RGBColor(0x33, 0x33, 0x55))

add_bullet_frame(slide2, Inches(7.1), Inches(1.7), Inches(5.1), Inches(5),
                 [
                     "Manual ECG reading is slow, subjective, and requires years of specialized training.",
                     "Existing commercial solutions (Apple Watch, AliveCor) detect only atrial fibrillation - not the full spectrum of arrhythmias.",
                     "Cloud-based AI solutions introduce network latency - unacceptable for real-time critical care.",
                     "Most published AI models inflate accuracy by mixing beats from the same patient in train/test sets - producing misleading 99%+ claims.",
                     "No affordable, open-source, edge-deployable solution exists for comprehensive beat-level arrhythmia classification."
                 ],
                 title="Existing Gaps", font_size=15, color=LIGHT)


# =====================================================================
# SLIDE 3: PROPOSED SOLUTION
# =====================================================================
slide3 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide3, BG_DARK)
add_accent_line(slide3, Inches(0), Inches(0), SLIDE_W)

add_text_box(slide3, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "PROPOSED SOLUTION",
             font_size=32, color=WHITE, bold=True)
add_accent_line(slide3, Inches(0.8), Inches(1.1), Inches(3))

# Main description
add_text_box(slide3, Inches(0.8), Inches(1.4), Inches(11.7), Inches(0.8),
             "An AI-powered real-time ECG monitoring system that classifies every heartbeat into 5 clinical categories using a 1D Residual CNN trained on the gold-standard MIT-BIH Arrhythmia Database.",
             font_size=18, color=LIGHT)

# Feature cards row
features = [
    ("Real-Time\nClassification", "Classifies each heartbeat\nin milliseconds - fast enough\nfor live patient monitoring", ACCENT),
    ("5-Class\nDetection", "Normal (N), Supraventricular (S),\nVentricular (V), Fusion (F),\nUnclassifiable (Q)", GREEN),
    ("97.2% PVC\nRecall", "Catches 97.2% of dangerous\nVentricular beats - prioritizing\nsafety over false negatives", ACCENT2),
    ("Edge-Ready\nModel", "Lightweight 606K-parameter\nmodel runs on CPU - no GPU\nor cloud connection needed", ORANGE),
]

for i, (title, desc, clr) in enumerate(features):
    left = Inches(0.8 + i * 3.1)
    card = add_shape_rect(slide3, left, Inches(2.5), Inches(2.8), Inches(2.6), BG_CARD,
                          border_color=RGBColor(0x33, 0x33, 0x55))
    add_text_box(slide3, left + Inches(0.15), Inches(2.65), Inches(2.5), Inches(0.8),
                 title, font_size=20, color=clr, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide3, left + Inches(0.15), Inches(3.5), Inches(2.5), Inches(1.4),
                 desc, font_size=14, color=LIGHT, alignment=PP_ALIGN.CENTER)

# USP
add_shape_rect(slide3, Inches(0.8), Inches(5.4), Inches(11.7), Inches(1.4), BG_CARD,
               border_color=ACCENT)
add_text_box(slide3, Inches(1.1), Inches(5.5), Inches(11.1), Inches(0.5),
             "Unique Selling Proposition (USP)",
             font_size=18, color=ACCENT, bold=True)
add_text_box(slide3, Inches(1.1), Inches(6.0), Inches(11.1), Inches(0.7),
             "Unlike commercial wearables (Apple Watch detects only AFib) or cloud APIs (latency issues), our solution provides comprehensive 5-class arrhythmia detection, runs entirely on-device with no internet required, and uses a scientifically rigorous patient-level data split for honest, reproducible accuracy metrics.",
             font_size=15, color=LIGHT)


# =====================================================================
# SLIDE 4: SOLUTION ARCHITECTURE - HOW IT WORKS
# =====================================================================
slide4 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide4, BG_DARK)
add_accent_line(slide4, Inches(0), Inches(0), SLIDE_W)

add_text_box(slide4, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "SOLUTION ARCHITECTURE - HOW IT WORKS",
             font_size=32, color=WHITE, bold=True)
add_accent_line(slide4, Inches(0.8), Inches(1.1), Inches(3))

# Pipeline steps
steps = [
    ("1. DATA\nACQUISITION", "MIT-BIH Arrhythmia\nDatabase (48 patients)\nfrom PhysioNet", ACCENT),
    ("2. PREPROCESSING", "R-peak detection\nBeat segmentation\n(250 samples/beat)\nZ-score normalization", GREEN),
    ("3. AUGMENTATION", "Random oversampling\nof minority classes\n(S, V, F, Q)\nto 50% of majority", ORANGE),
    ("4. AI MODEL", "1D Residual CNN\n(3 residual blocks)\n64->128->256 filters\nFocal Loss training", ACCENT2),
    ("5. INFERENCE", "Real-time beat\nclassification in <5ms\nper heartbeat on CPU", ACCENT),
]

for i, (title, desc, clr) in enumerate(steps):
    left = Inches(0.5 + i * 2.5)
    # Step box
    card = add_shape_rect(slide4, left, Inches(1.6), Inches(2.2), Inches(2.8), BG_CARD,
                          border_color=RGBColor(0x33, 0x33, 0x55))
    add_text_box(slide4, left + Inches(0.1), Inches(1.7), Inches(2.0), Inches(0.8),
                 title, font_size=16, color=clr, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide4, left + Inches(0.1), Inches(2.5), Inches(2.0), Inches(1.6),
                 desc, font_size=13, color=LIGHT, alignment=PP_ALIGN.CENTER)

    # Arrow between steps
    if i < len(steps) - 1:
        add_text_box(slide4, left + Inches(2.2), Inches(2.5), Inches(0.3), Inches(0.5),
                     "\u27A4", font_size=20, color=ACCENT, alignment=PP_ALIGN.CENTER)

# Model architecture detail
add_shape_rect(slide4, Inches(0.8), Inches(4.8), Inches(11.7), Inches(2.2), BG_CARD,
               border_color=RGBColor(0x33, 0x33, 0x55))
add_text_box(slide4, Inches(1.1), Inches(4.9), Inches(5), Inches(0.5),
             "Residual CNN Architecture", font_size=18, color=ACCENT, bold=True)

arch_text = (
    "Input (250,1) -> Conv1D(64) + BN + ReLU -> ResBlock(64) -> ResBlock(128) -> ResBlock(256)\n"
    "-> GlobalAvgPool -> Dense(128) + Dropout(0.5) -> Softmax(5 classes)\n\n"
    "Each ResBlock: Conv1D -> BN -> ReLU -> Conv1D -> BN + Skip Connection -> ReLU -> MaxPool"
)
add_text_box(slide4, Inches(1.1), Inches(5.4), Inches(5.5), Inches(1.4),
             arch_text, font_size=13, color=LIGHT)

# Training details
add_text_box(slide4, Inches(7), Inches(4.9), Inches(5), Inches(0.5),
             "Training Configuration", font_size=18, color=ACCENT, bold=True)
train_text = (
    "Loss: Focal Loss (gamma=2.0) - auto-focuses on hard/rare samples\n"
    "Optimizer: Adam (LR=0.001 with ReduceLROnPlateau)\n"
    "Early Stopping: patience=15 on val_accuracy\n"
    "Training: 86,739 beats from 39 patients (oversampled to 213K)\n"
    "Test: 11,738 beats from 5 completely unseen patients"
)
add_text_box(slide4, Inches(7), Inches(5.4), Inches(5), Inches(1.4),
             train_text, font_size=13, color=LIGHT)


# =====================================================================
# SLIDE 5: AI INNOVATION & TECH STACK
# =====================================================================
slide5 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide5, BG_DARK)
add_accent_line(slide5, Inches(0), Inches(0), SLIDE_W)

add_text_box(slide5, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "AI INNOVATION & TECH STACK",
             font_size=32, color=WHITE, bold=True)
add_accent_line(slide5, Inches(0.8), Inches(1.1), Inches(3))

# Left: AI Innovation
add_shape_rect(slide5, Inches(0.8), Inches(1.5), Inches(6), Inches(5.3), BG_CARD,
               border_color=RGBColor(0x33, 0x33, 0x55))

innovations = [
    "Residual Skip Connections: Prevent gradient vanishing in deeper networks, enabling the model to learn complex morphological patterns across heartbeat waveforms.",
    "Focal Loss (gamma=2.0): Automatically down-weights easy Normal beats and focuses training on rare arrhythmias (S, V, F, Q) without manual class weight tuning.",
    "Patient-Level Data Split: Training and test patients are strictly separated. The AI has never seen the test patients' hearts, proving real generalization - not memorization.",
    "Batch Normalization + Dropout: Regularization techniques that prevent overfitting on the 86K training beats, ensuring the model transfers to unseen patients.",
]
add_bullet_frame(slide5, Inches(1.1), Inches(1.7), Inches(5.4), Inches(5),
                 innovations, title="AI Innovations", font_size=14, color=LIGHT)

# Right: Tech Stack
add_shape_rect(slide5, Inches(7.2), Inches(1.5), Inches(5.3), Inches(5.3), BG_CARD,
               border_color=RGBColor(0x33, 0x33, 0x55))

add_text_box(slide5, Inches(7.5), Inches(1.7), Inches(4.7), Inches(0.5),
             "Tech Stack", font_size=22, color=ACCENT, bold=True)

stack_items = [
    ("Deep Learning", "TensorFlow 2.21 / Keras"),
    ("Architecture", "1D Residual CNN (606K params)"),
    ("Loss Function", "Custom Focal Loss"),
    ("Dataset", "MIT-BIH Arrhythmia Database (PhysioNet)"),
    ("Signal Processing", "wfdb, NumPy, SciPy"),
    ("Dashboard", "Streamlit (real-time web UI)"),
    ("Visualization", "Plotly (interactive ECG charts)"),
    ("Language", "Python 3.12"),
    ("Deployment", "CPU-only, edge-compatible"),
]

y_pos = Inches(2.3)
for label, value in stack_items:
    add_text_box(slide5, Inches(7.5), y_pos, Inches(2.2), Inches(0.35),
                 label, font_size=13, color=ACCENT, bold=True)
    add_text_box(slide5, Inches(9.5), y_pos, Inches(2.7), Inches(0.35),
                 value, font_size=13, color=LIGHT)
    y_pos += Inches(0.4)


# =====================================================================
# SLIDE 6: FEASIBILITY, VIABILITY & SCALABILITY
# =====================================================================
slide6 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide6, BG_DARK)
add_accent_line(slide6, Inches(0), Inches(0), SLIDE_W)

add_text_box(slide6, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "FEASIBILITY, VIABILITY & SCALABILITY",
             font_size=32, color=WHITE, bold=True)
add_accent_line(slide6, Inches(0.8), Inches(1.1), Inches(3))

# Three columns
col_data = [
    ("Technical Feasibility", ACCENT, [
        "Fully working prototype demonstrated with real hospital ECG data.",
        "Runs on standard laptop CPU - no GPU or cloud infrastructure needed.",
        "Model is only 2.3 MB - fits on any edge device (Raspberry Pi, smartphone).",
        "Open-source dataset (MIT-BIH) - no data licensing issues.",
        "All dependencies are free, open-source Python libraries.",
    ]),
    ("Business Viability", GREEN, [
        "Hospital ICU monitoring: reduce alarm fatigue by filtering false alerts.",
        "Telemedicine: enable remote cardiac monitoring for rural patients.",
        "Wearable integration: embed into smartwatch or holter monitor firmware.",
        "Cost: Near-zero marginal cost per prediction (CPU inference).",
        "Revenue model: SaaS for hospitals, licensing for device manufacturers.",
    ]),
    ("Scalability", ORANGE, [
        "Horizontal: Add more arrhythmia classes (12+ types) with more training data.",
        "Multi-lead: Extend from single-lead (MLII) to 12-lead ECG analysis.",
        "Multi-device: Deploy on smartphone, Raspberry Pi, or hospital workstations.",
        "Cloud option: Batch processing of stored ECG recordings for retrospective analysis.",
        "Global: Multilingual dashboard for international hospital deployment.",
    ]),
]

for i, (title, clr, items) in enumerate(col_data):
    left = Inches(0.8 + i * 4.1)
    add_shape_rect(slide6, left, Inches(1.5), Inches(3.8), Inches(5.3), BG_CARD,
                   border_color=RGBColor(0x33, 0x33, 0x55))
    add_bullet_frame(slide6, left + Inches(0.2), Inches(1.7), Inches(3.4), Inches(5),
                     items, title=title, font_size=13, color=LIGHT, title_size=18)


# =====================================================================
# SLIDE 7: IMPACT & FUTURE SCOPE
# =====================================================================
slide7 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide7, BG_DARK)
add_accent_line(slide7, Inches(0), Inches(0), SLIDE_W)

add_text_box(slide7, Inches(0.8), Inches(0.4), Inches(11), Inches(0.7),
             "IMPACT & FUTURE SCOPE",
             font_size=32, color=WHITE, bold=True)
add_accent_line(slide7, Inches(0.8), Inches(1.1), Inches(3))

# Metric cards row
metrics = [
    ("97.2%", "Ventricular Beat\nRecall Rate", ACCENT2),
    ("81.1%", "Overall Test\nAccuracy", ACCENT),
    ("86,739", "Training\nBeats", GREEN),
    ("<5ms", "Inference\nTime per Beat", ORANGE),
]

for i, (val, label, clr) in enumerate(metrics):
    add_metric_card(slide7, Inches(0.8 + i * 3.1), Inches(1.4), Inches(2.8), Inches(1.2),
                    val, label, value_color=clr)

# Impact
add_shape_rect(slide7, Inches(0.8), Inches(3.0), Inches(5.8), Inches(3.9), BG_CARD,
               border_color=RGBColor(0x33, 0x33, 0x55))
add_bullet_frame(slide7, Inches(1.1), Inches(3.2), Inches(5.2), Inches(3.5),
                 [
                     "Social: Early arrhythmia detection can prevent sudden cardiac deaths - potentially saving millions of lives in under-resourced regions.",
                     "Economic: Reduce ICU monitoring costs by automating first-pass ECG screening. One AI system can monitor multiple patients simultaneously.",
                     "Healthcare Access: Enable cardiac monitoring in rural areas without specialist cardiologists through telemedicine integration.",
                     "Research: Open-source codebase enables reproducible cardiac AI research and education.",
                 ],
                 title="Expected Impact", font_size=14, color=LIGHT)

# Future Scope
add_shape_rect(slide7, Inches(7.0), Inches(3.0), Inches(5.5), Inches(3.9), BG_CARD,
               border_color=RGBColor(0x33, 0x33, 0x55))
add_bullet_frame(slide7, Inches(7.3), Inches(3.2), Inches(4.9), Inches(3.5),
                 [
                     "12-Lead ECG Support: Extend to full clinical-grade 12-lead analysis for comprehensive cardiac assessment.",
                     "Transformer Architecture: Explore attention-based models for better temporal pattern recognition.",
                     "Continuous Learning: Implement federated learning to improve the model across multiple hospitals without sharing patient data.",
                     "FDA Clearance Path: Work toward 510(k) regulatory approval for clinical deployment.",
                     "Wearable SDK: Package as an SDK for smartwatch/fitness band manufacturers.",
                 ],
                 title="Future Roadmap", font_size=14, color=LIGHT)


# =====================================================================
# SLIDE 8: THANK YOU
# =====================================================================
slide8 = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide8, BG_DARK)
add_accent_line(slide8, Inches(0), Inches(0), SLIDE_W)

add_text_box(slide8, Inches(2), Inches(1.5), Inches(9.3), Inches(1),
             "\u2764", font_size=60, color=ACCENT2, alignment=PP_ALIGN.CENTER)

add_text_box(slide8, Inches(2), Inches(2.5), Inches(9.3), Inches(1),
             "THANK YOU!",
             font_size=52, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

add_accent_line(slide8, Inches(5.5), Inches(3.6), Inches(2.3))

add_text_box(slide8, Inches(2), Inches(4.0), Inches(9.3), Inches(0.6),
             "Real-Time AI ECG Arrhythmia Detector",
             font_size=24, color=ACCENT, alignment=PP_ALIGN.CENTER)

add_text_box(slide8, Inches(2), Inches(4.7), Inches(9.3), Inches(0.5),
             "Team: [Your Team Name]",
             font_size=20, color=MUTED, alignment=PP_ALIGN.CENTER)

add_text_box(slide8, Inches(2), Inches(5.3), Inches(9.3), Inches(0.5),
             "GitHub: github.com/[your-username]/ecg-arrhythmia-detector",
             font_size=16, color=MUTED, alignment=PP_ALIGN.CENTER)

add_text_box(slide8, Inches(2), Inches(6.2), Inches(9.3), Inches(0.5),
             "Questions?",
             font_size=28, color=LIGHT, alignment=PP_ALIGN.CENTER)


# ========== SAVE ==========
prs.save(OUTPUT_FILE)
print(f"Presentation saved to: {OUTPUT_FILE}")
print(f"Total slides: {len(prs.slides)}")
