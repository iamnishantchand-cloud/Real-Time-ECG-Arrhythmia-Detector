"""
Generate a beginner-friendly Project Explanation PPT for the ECG Arrhythmia Detector.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Colors ──
BG_DARK = RGBColor(0x0F, 0x17, 0x2A)
BG_CARD = RGBColor(0x1A, 0x25, 0x3C)
TEAL = RGBColor(0x00, 0xD2, 0xD3)
RED = RGBColor(0xEE, 0x55, 0x55)
GREEN = RGBColor(0x00, 0xE6, 0x76)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xBB, 0xBB, 0xCC)
YELLOW = RGBColor(0xFF, 0xD9, 0x3D)
ORANGE = RGBColor(0xFF, 0xA5, 0x02)
PURPLE = RGBColor(0xBB, 0x86, 0xFC)


def set_bg(slide, color=BG_DARK):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=14,
                 color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Segoe UI"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
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


def add_para(text_frame, text, font_size=13, color=LIGHT_GRAY, bold=False, space_before=Pt(6)):
    p = text_frame.add_paragraph()
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = "Segoe UI"
    p.space_before = space_before
    return p


def add_card(slide, left, top, width, height, color=BG_CARD):
    shape = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))  # 1 = rectangle
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


# ═══════════════════════════════════════════════════════════════
# SLIDE 1: TITLE
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
set_bg(slide)
add_text_box(slide, 1, 1.5, 11, 1, "Real-Time AI ECG Arrhythmia Detector", 40, TEAL, True, PP_ALIGN.CENTER)
add_text_box(slide, 1, 2.7, 11, 0.8, "Complete Project Explanation", 24, WHITE, False, PP_ALIGN.CENTER)
add_text_box(slide, 1, 3.8, 11, 0.6, "A beginner-friendly guide to every aspect of this project", 16, LIGHT_GRAY, False, PP_ALIGN.CENTER)
add_text_box(slide, 1, 5.5, 11, 0.5, "No prior knowledge of AI, medicine, or programming is assumed", 14, YELLOW, False, PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════
# SLIDE 2: TABLE OF CONTENTS
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "What You'll Learn", 32, TEAL, True)

topics_left = [
    "1.  What Problem Does This Solve?",
    "2.  What is an ECG?",
    "3.  What is an Arrhythmia?",
    "4.  The 5 Types of Heartbeats We Detect",
    "5.  Where Does the Data Come From?",
    "6.  How Data is Prepared (data_prep.py)",
]
topics_right = [
    "7.   How the AI Model Works (train_model.py)",
    "8.   The Dual-Input Architecture",
    "9.   How the Dashboard Works (app.py)",
    "10.  Noise Robustness Testing",
    "11.  External Validation (INCART)",
    "12.  Key Results & What They Mean",
]

add_card(slide, 0.5, 1.2, 5.8, 5.5)
tb = add_text_box(slide, 0.8, 1.4, 5.3, 5, topics_left[0], 17, WHITE, False)
for t in topics_left[1:]:
    add_para(tb.text_frame, t, 17, WHITE, space_before=Pt(14))

add_card(slide, 6.8, 1.2, 5.8, 5.5)
tb = add_text_box(slide, 7.1, 1.4, 5.3, 5, topics_right[0], 17, WHITE, False)
for t in topics_right[1:]:
    add_para(tb.text_frame, t, 17, WHITE, space_before=Pt(14))

# ═══════════════════════════════════════════════════════════════
# SLIDE 3: THE PROBLEM
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "1. What Problem Does This Solve?", 32, TEAL, True)

add_card(slide, 0.5, 1.2, 12.3, 2.5)
tb = add_text_box(slide, 0.8, 1.4, 11.7, 2.2,
    "Imagine you're a nurse in a hospital ICU. You're watching 8 patient screens simultaneously. "
    "Each screen shows a squiggly green line - the patient's heartbeat. Your job is to spot the "
    "ONE dangerous heartbeat among thousands of normal ones.", 16, WHITE)
add_para(tb.text_frame, "", 8)
add_para(tb.text_frame, "Miss it, and the patient could die.", 18, RED, True)
add_para(tb.text_frame, "", 8)
add_para(tb.text_frame, 'Hospitals report that nurses ignore up to 90% of alarms due to "alarm fatigue" - too many false alerts.', 15, LIGHT_GRAY)

add_card(slide, 0.5, 4.0, 12.3, 2.8, RGBColor(0x0A, 0x2A, 0x1A))
tb = add_text_box(slide, 0.8, 4.2, 11.7, 2.5, "OUR SOLUTION", 14, GREEN, True)
add_para(tb.text_frame, "An AI system that watches every single heartbeat, classifies it instantly (< 5 milliseconds), "
    "and only alerts doctors when something is genuinely wrong.", 16, WHITE, space_before=Pt(10))
add_para(tb.text_frame, "", 8)
add_para(tb.text_frame, "It never gets tired. Never gets distracted. Catches 98.8% of dangerous heartbeats.", 17, GREEN, True)

# ═══════════════════════════════════════════════════════════════
# SLIDE 4: WHAT IS AN ECG
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "2. What is an ECG?", 32, TEAL, True)

add_card(slide, 0.5, 1.2, 6, 3)
tb = add_text_box(slide, 0.8, 1.4, 5.5, 2.7,
    "An ECG (Electrocardiogram) records the electrical activity of your heart.", 15, WHITE, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Every time your heart beats, an electrical signal travels through it, making the heart muscles contract and pump blood.", 14, LIGHT_GRAY)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "An ECG machine captures these signals through small sticky patches (electrodes) on your chest. The result is the familiar 'heartbeat line' on hospital monitors.", 14, LIGHT_GRAY)

add_card(slide, 6.8, 1.2, 6, 3)
tb = add_text_box(slide, 7.1, 1.3, 5.5, 0.4, "The PQRST Wave", 18, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Each heartbeat creates a specific pattern:", 14, LIGHT_GRAY)
add_para(tb.text_frame, "P wave - Upper chambers contract", 13, WHITE, space_before=Pt(8))
add_para(tb.text_frame, "QRS complex - Main chambers contract (big spike)", 13, WHITE)
add_para(tb.text_frame, "T wave - Heart recovers for next beat", 13, WHITE)
add_para(tb.text_frame, "R-peak - The tallest point, marks each beat", 13, TEAL, True)

add_card(slide, 0.5, 4.5, 12.3, 2.3)
tb = add_text_box(slide, 0.8, 4.6, 11.7, 0.4, "Key Numbers", 16, YELLOW, True)
add_para(tb.text_frame, "Signal measured in millivolts (mV) - tiny electrical voltages", 14, LIGHT_GRAY, space_before=Pt(10))
add_para(tb.text_frame, "MIT-BIH records at 360 samples per second (360 Hz) - 360 measurements every second", 14, LIGHT_GRAY)
add_para(tb.text_frame, "A 30-minute recording = 648,000 data points per patient", 14, LIGHT_GRAY)
add_para(tb.text_frame, "Our AI processes each heartbeat as a window of 250 samples (~0.7 seconds)", 14, WHITE, True)

# ═══════════════════════════════════════════════════════════════
# SLIDE 5: WHAT IS AN ARRHYTHMIA
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "3. What is an Arrhythmia?", 32, TEAL, True)

add_card(slide, 0.5, 1.2, 12.3, 2)
tb = add_text_box(slide, 0.8, 1.4, 11.7, 1.7,
    "An arrhythmia is any heartbeat that doesn't follow the normal rhythm.", 16, WHITE, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Think of your heart like a drummer keeping a steady beat. An arrhythmia is like the drummer suddenly hitting an extra beat, skipping a beat, or beating in a weird pattern.", 15, LIGHT_GRAY)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Some arrhythmias are harmless (everyone has occasional skipped beats). Others can be life-threatening.", 15, LIGHT_GRAY)

add_card(slide, 0.5, 3.5, 12.3, 3.5)
tb = add_text_box(slide, 0.8, 3.6, 11.7, 0.4, "The 5 Types We Detect (AAMI Standard)", 18, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "N - Normal Beat:  Heart beating normally from the right place  [SAFE]", 14, GREEN, space_before=Pt(8))
add_para(tb.text_frame, "S - Supraventricular:  Beat from the upper chambers instead of the normal pacemaker  [MODERATE]", 14, ORANGE)
add_para(tb.text_frame, "V - Ventricular (PVC):  Beat from the lower chambers - the WRONG place  [DANGEROUS]", 14, RED, True)
add_para(tb.text_frame, "F - Fusion:  A normal beat and ventricular beat happen simultaneously  [CONCERNING]", 14, PURPLE)
add_para(tb.text_frame, "Q - Unknown/Paced:  Unclassifiable beat, or from an artificial pacemaker  [DEPENDS]", 14, LIGHT_GRAY)
add_para(tb.text_frame, "", 8)
add_para(tb.text_frame, "PVCs are the most clinically dangerous. If the abnormal electrical source wins too often, it can cause ventricular fibrillation (heart quivers instead of pumping) - fatal within minutes.", 14, RED)

# ═══════════════════════════════════════════════════════════════
# SLIDE 6: WHERE DATA COMES FROM
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "4. Where Does the Data Come From?", 32, TEAL, True)

add_card(slide, 0.5, 1.2, 5.8, 3)
tb = add_text_box(slide, 0.8, 1.3, 5.3, 0.4, "MIT-BIH Arrhythmia Database", 18, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "The gold standard for ECG research since 1980", 14, WHITE, True)
add_para(tb.text_frame, "48 patient recordings, 30 minutes each", 14, LIGHT_GRAY, space_before=Pt(8))
add_para(tb.text_frame, "Recorded at 360 Hz (360 samples/second)", 14, LIGHT_GRAY)
add_para(tb.text_frame, "Every beat manually labeled by 2 cardiologists", 14, LIGHT_GRAY)
add_para(tb.text_frame, "Used in 5,000+ research papers", 14, LIGHT_GRAY)
add_para(tb.text_frame, "Free from PhysioNet (no licensing issues)", 14, LIGHT_GRAY)

add_card(slide, 6.8, 1.2, 5.8, 3)
tb = add_text_box(slide, 7.1, 1.3, 5.3, 0.4, "INCART Database (External Test)", 18, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Used to prove our model generalizes", 14, WHITE, True)
add_para(tb.text_frame, "75 patients from St. Petersburg, Russia", 14, LIGHT_GRAY, space_before=Pt(8))
add_para(tb.text_frame, "Different hospital, different equipment", 14, LIGHT_GRAY)
add_para(tb.text_frame, "173,476 beats - completely unseen data", 14, LIGHT_GRAY)
add_para(tb.text_frame, "Different sampling rate (257 Hz vs 360 Hz)", 14, LIGHT_GRAY)

add_card(slide, 0.5, 4.5, 12.3, 2.5, RGBColor(0x2A, 0x15, 0x0A))
tb = add_text_box(slide, 0.8, 4.6, 11.7, 0.4, "WHY PATIENT-LEVEL SPLITTING MATTERS", 16, RED, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Training: 43 patients  |  Testing: 5 completely separate patients", 15, WHITE, True, space_before=Pt(8))
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "The AI has NEVER seen the test patients' hearts during training. This is critical because:", 14, LIGHT_GRAY)
add_para(tb.text_frame, "If you mix beats from the same patient in train and test, the AI memorizes the patient's unique heart shape - not the arrhythmia. Many papers do this and report misleading 99%+ accuracy. Our 82% on truly unseen patients is more honest and more useful.", 14, LIGHT_GRAY)

# ═══════════════════════════════════════════════════════════════
# SLIDE 7: DATA PREPARATION
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "5. How Data is Prepared (data_prep.py)", 32, TEAL, True)
add_text_box(slide, 0.5, 0.9, 12, 0.4, "This is the FIRST script you run - it downloads and processes the raw ECG recordings", 14, LIGHT_GRAY)

steps = [
    ("Step 1: Download", "Download 48 patient recordings from PhysioNet (free medical data repository)"),
    ("Step 2: Segment Beats", "Find each R-peak (heartbeat marker) and cut a 250-sample window around it: 100 samples before + 150 after. Each window = 1 heartbeat."),
    ("Step 3: Normalize", "Scale each beat to 0-1 range so the AI focuses on SHAPE, not amplitude. Different patients have different signal strengths."),
    ("Step 4: RR Features", "Compute 4 timing features per beat: pre_RR (time since last beat), post_RR (time to next), local_avg_RR (average nearby), prematurity ratio."),
    ("Step 5: Label", "Map cardiologist annotations to 5 AAMI classes: N, S, V, F, Q."),
    ("Step 6: Save", "Save as .npy files: X_train (97,714 beats), y_train (labels), RR_train (timing features), plus test set and demo recordings."),
]

for i, (title, desc) in enumerate(steps):
    row = i // 3
    col = i % 3
    x = 0.5 + col * 4.2
    y = 1.5 + row * 2.8
    add_card(slide, x, y, 3.9, 2.5)
    add_text_box(slide, x + 0.2, y + 0.15, 3.5, 0.4, title, 15, TEAL, True)
    add_text_box(slide, x + 0.2, y + 0.6, 3.5, 1.8, desc, 12, LIGHT_GRAY)

# ═══════════════════════════════════════════════════════════════
# SLIDE 8: HOW AI MODEL WORKS
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "6. How the AI Model Works (train_model.py)", 32, TEAL, True)

add_card(slide, 0.5, 1.2, 6, 2.5)
tb = add_text_box(slide, 0.8, 1.3, 5.5, 0.4, "What is a Neural Network?", 18, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "A neural network is like a complex filter. You show it thousands of examples with correct answers, and it slowly adjusts its internal settings until it can predict the right answer on its own.", 13, LIGHT_GRAY, space_before=Pt(8))
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Like teaching a child dog breeds with photos - after enough examples, they recognize breeds in new photos they've never seen.", 13, WHITE)

add_card(slide, 6.8, 1.2, 6, 2.5)
tb = add_text_box(slide, 7.1, 1.3, 5.5, 0.4, "What is a 1D-CNN?", 18, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "CNN = Convolutional Neural Network. A small 'sliding window' moves across the heartbeat signal, computing how much each part matches patterns it's looking for.", 13, LIGHT_GRAY, space_before=Pt(8))
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Layer 1: Detects simple features (spikes, slopes)\nLayer 2: Detects QRS patterns, T-wave shapes\nLayer 3: Detects overall beat type (Normal vs PVC)", 13, WHITE)

add_card(slide, 0.5, 4.0, 4, 3)
tb = add_text_box(slide, 0.8, 4.1, 3.5, 0.4, "Residual (Skip) Connections", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Deep networks suffer from 'vanishing gradients' - information gets diluted through many layers.", 12, LIGHT_GRAY, space_before=Pt(6))
add_para(tb.text_frame, "", 4)
add_para(tb.text_frame, "Skip connections add shortcuts that let information bypass layers. This allows deeper, more powerful networks.", 12, WHITE)
add_para(tb.text_frame, "", 4)
add_para(tb.text_frame, "Our model uses 3 residual blocks (64, 128, 256 filters).", 12, TEAL, True)

add_card(slide, 4.8, 4.0, 4, 3)
tb = add_text_box(slide, 5.1, 4.1, 3.5, 0.4, "Focal Loss", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "82% of beats are Normal. A lazy AI could predict 'Normal' always and be 82% accurate - but miss every arrhythmia!", 12, LIGHT_GRAY, space_before=Pt(6))
add_para(tb.text_frame, "", 4)
add_para(tb.text_frame, "Focal Loss forces the AI to focus on rare, hard cases. Easy Normal beats get tiny penalties. Missed PVCs get HUGE penalties.", 12, WHITE)
add_para(tb.text_frame, "", 4)
add_para(tb.text_frame, "Like a teacher who spends more time on topics students struggle with.", 12, TEAL)

add_card(slide, 9.1, 4.0, 3.7, 3)
tb = add_text_box(slide, 9.4, 4.1, 3.2, 0.4, "Data Augmentation", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Rare classes have few examples. We create more by slightly modifying existing ones:", 12, LIGHT_GRAY, space_before=Pt(6))
add_para(tb.text_frame, "", 4)
add_para(tb.text_frame, "1. Add tiny random noise\n2. Scale amplitude (+/-15%)\n3. Shift timing (+/-5 samples)", 12, WHITE)
add_para(tb.text_frame, "", 4)
add_para(tb.text_frame, "Grows training set from 97K to 218K beats.", 12, TEAL, True)

# ═══════════════════════════════════════════════════════════════
# SLIDE 9: DUAL-INPUT ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "7. Dual-Input Architecture - Our Secret Weapon", 32, TEAL, True)

add_card(slide, 0.5, 1.2, 5.8, 2.5)
tb = add_text_box(slide, 0.8, 1.3, 5.3, 0.4, "The Problem", 18, RED, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Most ECG AI models look ONLY at the beat waveform shape.", 14, LIGHT_GRAY, space_before=Pt(8))
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "But cardiologists also look at RHYTHM - how the beat relates to its neighbors in time:", 14, WHITE)
add_para(tb.text_frame, "'This beat came too early' = might be premature (PVC)", 13, ORANGE, space_before=Pt(8))
add_para(tb.text_frame, "'Long pause after this beat' = compensatory pause (confirms PVC)", 13, ORANGE)

add_card(slide, 6.8, 1.2, 5.8, 2.5)
tb = add_text_box(slide, 7.1, 1.3, 5.3, 0.4, "Our Solution: Two Inputs", 18, GREEN, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Branch 1 (CNN): Learns what the beat LOOKS like (morphology/shape)", 14, WHITE, space_before=Pt(8))
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Branch 2 (Dense): Learns what the beat's TIMING tells us (rhythm)", 14, WHITE)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Together, they make better decisions - just like a cardiologist who considers both shape AND rhythm.", 14, GREEN)

add_card(slide, 0.5, 4.0, 12.3, 3.2)
tb = add_text_box(slide, 0.8, 4.1, 11.7, 0.4, "Architecture Diagram", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "INPUT 1: Beat Waveform (250 samples)", 14, TEAL, True, space_before=Pt(6))
add_para(tb.text_frame, "    Conv1D(64,7) -> ResBlock(64) -> ResBlock(128) -> ResBlock(256) -> GlobalAvgPool", 12, LIGHT_GRAY)
add_para(tb.text_frame, "                                                                            |", 12, LIGHT_GRAY)
add_para(tb.text_frame, "                                                                      CONCATENATE -> Dense(256) -> Dropout -> Dense(128) -> Softmax(5)", 12, WHITE, True)
add_para(tb.text_frame, "                                                                            |", 12, LIGHT_GRAY)
add_para(tb.text_frame, "INPUT 2: RR Features (4 numbers)", 14, TEAL, True)
add_para(tb.text_frame, "    Dense(16) -> BatchNorm -> Dense(8) ----------------------------------------+", 12, LIGHT_GRAY)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "RR features are patient-independent: PVC timing (early beat + pause) is universal regardless of waveform shape.", 13, GREEN)

# ═══════════════════════════════════════════════════════════════
# SLIDE 10: DASHBOARD
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "8. How the Dashboard Works (app.py)", 32, TEAL, True)

add_card(slide, 0.5, 1.2, 6, 3)
tb = add_text_box(slide, 0.8, 1.3, 5.5, 0.4, "Tab 1: Live Monitor", 18, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Simulates a real hospital bedside monitor:", 14, WHITE, space_before=Pt(6))
add_para(tb.text_frame, "1. Select a patient recording from dropdown", 13, LIGHT_GRAY, space_before=Pt(8))
add_para(tb.text_frame, "2. Click Play - ECG scrolls across screen", 13, LIGHT_GRAY)
add_para(tb.text_frame, "3. Each beat classified in real-time:", 13, LIGHT_GRAY)
add_para(tb.text_frame, "   Green = Normal | Red = PVC | Orange = SVE", 13, WHITE, True)
add_para(tb.text_frame, "4. Live heart rate + arrhythmia count", 13, LIGHT_GRAY)
add_para(tb.text_frame, "5. Alert log for every abnormal beat", 13, LIGHT_GRAY)

add_card(slide, 6.8, 1.2, 6, 3)
tb = add_text_box(slide, 7.1, 1.3, 5.5, 0.4, "Tab 2: Model Performance", 18, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Shows the AI's test results:", 14, WHITE, space_before=Pt(6))
add_para(tb.text_frame, "Confusion Matrix - predictions vs reality grid", 13, LIGHT_GRAY, space_before=Pt(8))
add_para(tb.text_frame, "Per-class precision, recall, F1-score", 13, LIGHT_GRAY)
add_para(tb.text_frame, "Training accuracy curves over epochs", 13, LIGHT_GRAY)
add_para(tb.text_frame, "Class distribution charts", 13, LIGHT_GRAY)

add_card(slide, 0.5, 4.5, 12.3, 2.5)
tb = add_text_box(slide, 0.8, 4.6, 11.7, 0.4, "How Inference Works (Behind the Scenes)", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "1. Load recording -> 2. Feed ALL beats + RR features through model at once (batch prediction) -> 3. Model outputs probabilities [P(N), P(S), P(V), P(F), P(Q)] -> 4. Cache results -> 5. During playback, just display cached results frame by frame", 14, LIGHT_GRAY, space_before=Pt(8))
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "KEY INSIGHT: All inference runs ONCE when you select a recording, not during playback. That's why animation is smooth.", 14, GREEN, True)

# ═══════════════════════════════════════════════════════════════
# SLIDE 11: NOISE ROBUSTNESS
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "9. Noise Robustness Testing", 32, TEAL, True)

add_card(slide, 0.5, 1.2, 6, 2)
tb = add_text_box(slide, 0.8, 1.3, 5.5, 0.4, "Why Test With Noise?", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "In real hospitals, ECG signals are never perfectly clean. Patients move, electrodes shift, machines interfere. A model that works on clean data but fails on noisy data is useless.", 14, LIGHT_GRAY)

add_card(slide, 6.8, 1.2, 6, 2)
tb = add_text_box(slide, 7.1, 1.3, 5.5, 0.4, "3 Types of Noise We Test", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Gaussian - Random static (like radio interference)", 13, LIGHT_GRAY, space_before=Pt(6))
add_para(tb.text_frame, "Baseline Wander - Slow drift from breathing/movement", 13, LIGHT_GRAY)
add_para(tb.text_frame, "Muscle Artifact - High-freq noise from fidgeting", 13, LIGHT_GRAY)

add_card(slide, 0.5, 3.5, 12.3, 3.7)
tb = add_text_box(slide, 0.8, 3.6, 11.7, 0.4, "Results: Accuracy at Different Noise Levels", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "SNR (dB)     Gaussian    Baseline Wander    Muscle Artifact    Combined", 13, TEAL, True, space_before=Pt(6))
add_para(tb.text_frame, "Clean          81.1%         81.1%              81.1%            81.1%", 13, WHITE, space_before=Pt(6))
add_para(tb.text_frame, "30 dB         79.3%         81.3%              80.3%            80.7%     (typical hospital)", 13, GREEN)
add_para(tb.text_frame, "20 dB         15.4%         81.6%              26.3%            25.1%", 13, LIGHT_GRAY)
add_para(tb.text_frame, "10 dB           2.1%         81.4%                2.0%              3.1%", 13, LIGHT_GRAY)
add_para(tb.text_frame, "  0 dB           1.3%         71.3%                9.7%              2.3%     (extreme noise)", 13, ORANGE)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Key finding: Model is very robust to baseline wander (71.3% even at 0dB!) - the most common real-world noise source.", 14, GREEN, True)

# ═══════════════════════════════════════════════════════════════
# SLIDE 12: EXTERNAL VALIDATION
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "10. External Validation (INCART Database)", 32, TEAL, True)

add_card(slide, 0.5, 1.2, 6, 2.5)
tb = add_text_box(slide, 0.8, 1.3, 5.5, 0.4, "Why External Validation?", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Training and testing on the same database is like studying for an exam using the exact textbook the exam comes from.", 14, LIGHT_GRAY, space_before=Pt(6))
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "We tested on INCART - a completely different hospital (Russia vs USA), different equipment, different patients. This proves real generalization.", 14, WHITE)

add_card(slide, 6.8, 1.2, 6, 2.5)
tb = add_text_box(slide, 7.1, 1.3, 5.5, 0.4, "INCART Results", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "87% overall accuracy on 173,476 beats", 16, GREEN, True, space_before=Pt(6))
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Normal beats: correctly identified across databases", 14, WHITE)
add_para(tb.text_frame, "Arrhythmia detection: limited (expected)", 14, LIGHT_GRAY)
add_para(tb.text_frame, "Shows the path forward: domain adaptation needed for multi-site deployment", 14, LIGHT_GRAY)

add_card(slide, 0.5, 4.0, 12.3, 2, RGBColor(0x0A, 0x2A, 0x1A))
tb = add_text_box(slide, 0.8, 4.2, 11.7, 1.5,
    "This result demonstrates scientific rigor. Most hackathon/research projects never test on external data. "
    "Showing this - even with imperfect results - proves we understand real-world deployment challenges and sets "
    "our project apart from those that only show inflated numbers on their training dataset.", 15, GREEN)

# ═══════════════════════════════════════════════════════════════
# SLIDE 13: KEY RESULTS
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "11. Key Results & What They Mean", 32, TEAL, True)

# Stat cards
stats = [
    ("98.8%", "PVC Recall", "Catches nearly every dangerous beat", RED),
    ("82.0%", "Test Accuracy", "On patients NEVER seen in training", TEAL),
    ("608K", "Parameters", "Lightweight model, runs anywhere", PURPLE),
    ("<5ms", "Inference Time", "Real-time on CPU, no GPU needed", GREEN),
]
for i, (val, label, desc, color) in enumerate(stats):
    x = 0.5 + i * 3.15
    add_card(slide, x, 1.2, 2.9, 2)
    add_text_box(slide, x + 0.2, 1.3, 2.5, 0.7, val, 36, color, True, PP_ALIGN.CENTER)
    add_text_box(slide, x + 0.2, 2.0, 2.5, 0.4, label, 14, WHITE, True, PP_ALIGN.CENTER)
    add_text_box(slide, x + 0.2, 2.4, 2.5, 0.5, desc, 11, LIGHT_GRAY, False, PP_ALIGN.CENTER)

add_card(slide, 0.5, 3.5, 6, 3.5)
tb = add_text_box(slide, 0.8, 3.6, 5.5, 0.4, "Precision vs Recall (Explained Simply)", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "RECALL = Of all actual PVCs, what % did we catch?", 14, WHITE, True, space_before=Pt(6))
add_para(tb.text_frame, "98.8% = we only miss 1.2% of dangerous beats", 13, GREEN)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "PRECISION = Of our PVC alerts, what % were real?", 14, WHITE, True)
add_para(tb.text_frame, "41.5% = some false alarms, but safe", 13, ORANGE)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "Think SMOKE DETECTOR: Better to have false alarms than miss a real fire. We prioritize HIGH RECALL for patient safety.", 13, YELLOW, True)

add_card(slide, 6.8, 3.5, 6, 3.5)
tb = add_text_box(slide, 7.1, 3.6, 5.5, 0.4, "Model Evolution", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "v1: Basic CNN", 14, LIGHT_GRAY, space_before=Pt(6))
add_para(tb.text_frame, "      67.3% accuracy | 96.8% V recall", 12, LIGHT_GRAY)
add_para(tb.text_frame, "v2: + Class Weights", 14, LIGHT_GRAY, space_before=Pt(8))
add_para(tb.text_frame, "      81.4% accuracy | 96.8% V recall", 12, LIGHT_GRAY)
add_para(tb.text_frame, "v3: + Residual CNN + Focal Loss", 14, WHITE, space_before=Pt(8))
add_para(tb.text_frame, "      81.1% accuracy | 97.2% V recall", 12, WHITE)
add_para(tb.text_frame, "v4: + Dual-Input + RR Features", 14, GREEN, True, space_before=Pt(8))
add_para(tb.text_frame, "      82.0% accuracy | 98.8% V recall", 12, GREEN, True)

# ═══════════════════════════════════════════════════════════════
# SLIDE 14: FILE STRUCTURE
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 0.5, 0.3, 12, 0.7, "12. How All Files Connect Together", 32, TEAL, True)

add_card(slide, 0.5, 1.2, 12.3, 2.5)
tb = add_text_box(slide, 0.8, 1.3, 11.7, 0.4, "Pipeline Flow", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "STEP 1: data_prep.py                    STEP 2: train_model.py                  STEP 3: app.py", 14, TEAL, True, space_before=Pt(6))
add_para(tb.text_frame, "Download from PhysioNet          Load .npy files                         Load trained model", 12, LIGHT_GRAY, space_before=Pt(6))
add_para(tb.text_frame, "Segment beats                          Oversample + Augment                 Live ECG playback", 12, LIGHT_GRAY)
add_para(tb.text_frame, "Normalize + RR features           Build Dual-Input CNN                  Real-time classification", 12, LIGHT_GRAY)
add_para(tb.text_frame, "Save .npy files            --->     Train with Focal Loss       --->     Display alerts", 12, WHITE, True)
add_para(tb.text_frame, "                                          Save .keras model                      Performance tab", 12, LIGHT_GRAY)

add_card(slide, 0.5, 4.0, 6, 3)
tb = add_text_box(slide, 0.8, 4.1, 5.5, 0.4, "Main Scripts", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "data_prep.py - Downloads & processes data (run FIRST)", 12, WHITE, space_before=Pt(6))
add_para(tb.text_frame, "train_model.py - Trains the AI model (run SECOND)", 12, WHITE)
add_para(tb.text_frame, "app.py - Dashboard for live demo (run ANYTIME)", 12, WHITE)
add_para(tb.text_frame, "noise_robustness.py - Tests noise resilience", 12, LIGHT_GRAY)
add_para(tb.text_frame, "external_validation.py - Tests on INCART database", 12, LIGHT_GRAY)
add_para(tb.text_frame, "create_ppt.py - Generates pitch deck", 12, LIGHT_GRAY)
add_para(tb.text_frame, "run_dashboard.bat - One-click launcher", 12, LIGHT_GRAY)

add_card(slide, 6.8, 4.0, 6, 3)
tb = add_text_box(slide, 7.1, 4.1, 5.5, 0.4, "Output Files (output/ folder)", 16, YELLOW, True)
add_para(tb.text_frame, "", 6)
add_para(tb.text_frame, "ecg_model.keras - Trained AI model (2.3 MB)", 12, WHITE, space_before=Pt(6))
add_para(tb.text_frame, "X_train.npy, y_train.npy - Training data", 12, LIGHT_GRAY)
add_para(tb.text_frame, "RR_train.npy, RR_test.npy - Timing features", 12, LIGHT_GRAY)
add_para(tb.text_frame, "metrics.json - Accuracy numbers", 12, LIGHT_GRAY)
add_para(tb.text_frame, "confusion_matrix.png - Prediction grid", 12, LIGHT_GRAY)
add_para(tb.text_frame, "noise_robustness_curve.png - Noise plot", 12, LIGHT_GRAY)
add_para(tb.text_frame, "demo_recordings/*.npz - Dashboard data", 12, LIGHT_GRAY)

# ═══════════════════════════════════════════════════════════════
# SLIDE 15: THANK YOU
# ═══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_text_box(slide, 1, 2, 11, 1, "Thank You!", 44, TEAL, True, PP_ALIGN.CENTER)
add_text_box(slide, 1, 3.2, 11, 0.6, "Real-Time AI ECG Arrhythmia Detector", 22, WHITE, False, PP_ALIGN.CENTER)
add_text_box(slide, 1, 4.2, 11, 0.5, "Now you understand every part of this project!", 16, LIGHT_GRAY, False, PP_ALIGN.CENTER)
add_text_box(slide, 1, 5.5, 11, 0.5, "Built with Python | TensorFlow | Streamlit | MIT-BIH Database", 14, YELLOW, False, PP_ALIGN.CENTER)

# ── Save ──
out = r"C:\Users\DELL\ecg-arrhythmia-detector\Project_Explanation_Presentation.pptx"
prs.save(out)
print(f"Saved: {out}")
print(f"Total slides: {len(prs.slides)}")
