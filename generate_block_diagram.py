import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = "output/report_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Create high-resolution publication figure
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.5)
ax.axis('off')

# Color Palette (Clean Medical / Engineering Theme)
C_INPUT  = "#e0f2fe" # Light Blue
B_INPUT  = "#0284c7"
C_BRANCH = "#dcfce7" # Light Green
B_BRANCH = "#16a34a"
C_FUSION = "#f3e8ff" # Light Purple
B_FUSION = "#9333ea"
C_OUTPUT = "#fee2e2" # Light Red
B_OUTPUT = "#dc2626"
C_SYSTEM = "#fef3c7" # Light Amber
B_SYSTEM = "#d97706"
ARROW_C  = "#475569"

def draw_box(x, y, w, h, title, subtitle, bg_color, border_color, title_size=9.5, sub_size=8):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                  ec=border_color, fc=bg_color, lw=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color="#0f172a", zorder=3)
    ax.text(x + w/2, y + h*0.30, subtitle, ha="center", va="center",
            fontsize=sub_size, color="#334155", zorder=3)
    return (x, y, w, h)

def draw_arrow(x1, y1, x2, y2, label=None, label_y_offset=0.25):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor=ARROW_C, edgecolor=ARROW_C,
                                arrowstyle="-|>", lw=1.8, mutation_scale=14),
                zorder=4)
    if label:
        ax.text((x1 + x2)/2, (y1 + y2)/2 + label_y_offset, label,
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1e293b", zorder=5)

# -------------------------------------------------------------
# COLUMN CONTAINERS (SUBGRAPHS)
# -------------------------------------------------------------
containers = [
    (0.4, 0.6, 2.3, 7.3, "1. MULTIMODAL INPUTS", "#f8fafc", "#cbd5e1"),
    (3.1, 0.6, 2.5, 7.3, "2. FEATURE EXTRACTION", "#f8fafc", "#cbd5e1"),
    (6.0, 1.8, 2.5, 5.0, "3. FUSION ENGINE", "#f8fafc", "#cbd5e1"),
    (8.9, 0.6, 2.4, 7.3, "4. DECISION & EXPLAINABILITY", "#f8fafc", "#cbd5e1"),
    (11.7, 1.8, 2.0, 5.0, "5. DEPLOYMENT", "#f8fafc", "#cbd5e1"),
]

for x, y, w, h, label, bg, bd in containers:
    c_rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                   ec=bd, fc=bg, lw=1.2, linestyle="--", zorder=1)
    ax.add_patch(c_rect)
    ax.text(x + w/2, y + h - 0.35, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#64748b", zorder=2)

# -------------------------------------------------------------
# 1. INPUT BLOCKS
# -------------------------------------------------------------
draw_box(0.6, 5.8, 1.9, 1.3, "Patient Image", "224 x 224 RGB\n(Posture / Facial Habit)", C_INPUT, B_INPUT)
draw_box(0.6, 3.6, 1.9, 1.3, "Speech Audio", "WAV / MP3 Waveform\n(Voice Recording)", C_INPUT, B_INPUT)
draw_box(0.6, 1.4, 1.9, 1.3, "Clinical Vitals", "10 Tabular Biomarkers\n(Age, BP, MoCA, Grip)", C_INPUT, B_INPUT)

# -------------------------------------------------------------
# 2. FEATURE EXTRACTION BLOCKS
# -------------------------------------------------------------
draw_box(3.3, 5.8, 2.1, 1.3, "Visual CNN Backbone", "GeriatricNet / ResNet50\nOutput: [128-d Vector]", C_BRANCH, B_BRANCH)
draw_box(3.3, 3.6, 2.1, 1.3, "Audio 2D-CNN Branch", "Librosa Mel-Spectrogram\nOutput: [32-d Vector]", C_BRANCH, B_BRANCH)
draw_box(3.3, 1.4, 2.1, 1.3, "Clinical ANN Branch", "Standard Dense (64->32)\nOutput: [32-d Vector]", C_BRANCH, B_BRANCH)

# Arrows from Input -> Extraction
draw_arrow(2.5, 6.45, 3.3, 6.45)
draw_arrow(2.5, 4.25, 3.3, 4.25)
draw_arrow(2.5, 2.05, 3.3, 2.05)

# -------------------------------------------------------------
# 3. FUSION BLOCKS
# -------------------------------------------------------------
draw_box(6.2, 4.8, 2.1, 1.3, "Intermediate Concatenate", "layers.Concatenate()\nUnified [192-d Vector]", C_FUSION, B_FUSION)
draw_box(6.2, 2.5, 2.1, 1.5, "Deep Dense Fusion", "Dense(256) -> BN -> Dropout(0.4)\nDense(128) -> BN -> Dropout(0.3)", C_FUSION, B_FUSION)

# Arrows from Extraction -> Fusion Concatenate
draw_arrow(5.4, 6.45, 6.2, 5.7)
draw_arrow(5.4, 4.25, 6.2, 5.2)
draw_arrow(5.4, 2.05, 6.2, 4.9)

# Arrow from Concatenate -> Deep Dense
draw_arrow(7.25, 4.8, 7.25, 4.0)

# -------------------------------------------------------------
# 4. DECISION & EXPLAINABILITY BLOCKS
# -------------------------------------------------------------
draw_box(9.1, 4.4, 2.0, 2.7, "Softmax Multi-Class\nClassifier Head", "1. Frailty\n2. Sarcopenia\n3. Cognitive Impairment\n4. Fall Risk\n5. Depression", C_OUTPUT, B_OUTPUT, title_size=9, sub_size=7.5)
draw_box(9.1, 1.4, 2.0, 1.8, "Grad-CAM Explainable AI", "Feature Map Gradients (gn_pw)\nVisual Saliency Heatmap\nClinician Audit Overlay", C_OUTPUT, B_OUTPUT, title_size=9, sub_size=7.5)

# Arrows from Fusion -> Outputs
draw_arrow(8.3, 3.25, 9.1, 5.5, label="Syndrome Risk")
draw_arrow(5.4, 6.45, 9.1, 2.3, label="CNN Gradients", label_y_offset=0.2)

# -------------------------------------------------------------
# 5. DEPLOYMENT BLOCKS
# -------------------------------------------------------------
draw_box(11.9, 4.6, 1.6, 1.5, "FastAPI Backend", "Asynchronous Uvicorn\nREST API (Port 8000)\nModel Cache Loader", C_SYSTEM, B_SYSTEM, title_size=8.5, sub_size=7)
draw_box(11.9, 2.2, 1.6, 1.5, "React Vite UI", "Interactive Dashboard\nRadar & Bar Charts\nEvidence Action Cards", C_SYSTEM, B_SYSTEM, title_size=8.5, sub_size=7)

# Arrows to Deployment
draw_arrow(11.1, 5.75, 11.9, 5.35)
draw_arrow(11.1, 2.3, 11.9, 2.7)
draw_arrow(12.7, 4.6, 12.7, 3.7, label="JSON / B64", label_y_offset=0.0)

# Overall Diagram Header
ax.text(7.0, 8.1, "GeroFusion Net: Complete End-to-End System Block Diagram",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")
ax.text(7.0, 7.75, "Multimodal Input Processing, Intermediate Feature-Level Fusion, Decision Head, Explainable AI, and Clinical Deployment",
        ha="center", va="center", fontsize=9.5, color="#475569")

plt.tight_layout()
out_file = os.path.join(OUT_DIR, "Figure_Slide_6_Block_Diagram_Entire_Project.png")
fig.savefig(out_file, bbox_inches='tight')
plt.close(fig)
print(f"Block Diagram successfully generated: {out_file}")
