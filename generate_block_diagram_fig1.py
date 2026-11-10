import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

OUT_DIR = "output/report_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Create high-resolution figure (15 x 8.5 inches, 300 DPI)
fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 9.0)
ax.axis('off')

# Color Palette (Publication Grade: Clean Slate / Medical Blue / Emerald / Purple)
C_INPUT  = "#e0f2fe"  # Light Blue
B_INPUT  = "#0284c7"
C_BRANCH = "#dcfce7"  # Light Green
B_BRANCH = "#16a34a"
C_FUSION = "#f3e8ff"  # Light Purple
B_FUSION = "#9333ea"
C_OUTPUT = "#fee2e2"  # Light Red
B_OUTPUT = "#dc2626"
C_SYSTEM = "#fef3c7"  # Light Amber
B_SYSTEM = "#d97706"
ARROW_C  = "#334155"

def draw_box(x, y, w, h, title, subtitle, bg_color, border_color, title_size=9, sub_size=7.5):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                  ec=border_color, fc=bg_color, lw=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color="#0f172a", zorder=4)
    ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center",
            fontsize=sub_size, color="#334155", zorder=4)
    return (x, y, w, h)

def draw_arrow(x1, y1, x2, y2, label=None, label_y_offset=0.22):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor=ARROW_C, edgecolor=ARROW_C,
                                arrowstyle="-|>", lw=1.8, mutation_scale=13),
                zorder=5)
    if label:
        ax.text((x1 + x2)/2, (y1 + y2)/2 + label_y_offset, label,
                ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1e293b", zorder=6)

# -------------------------------------------------------------
# COLUMN CONTAINERS (SUBGRAPHS)
# -------------------------------------------------------------
containers = [
    (0.4, 0.4, 2.3, 7.8, "1. MULTIMODAL INPUTS", "#f8fafc", "#cbd5e1"),
    (3.1, 0.4, 2.6, 7.8, "2. FEATURE EXTRACTION", "#f8fafc", "#cbd5e1"),
    (6.1, 1.5, 2.7, 5.7, "3. MULTIMODAL FUSION", "#f8fafc", "#cbd5e1"),
    (9.2, 0.4, 2.7, 7.8, "4. DECISION & EXPLAINABILITY", "#f8fafc", "#cbd5e1"),
    (12.3, 1.5, 2.3, 5.7, "5. DEPLOYMENT", "#f8fafc", "#cbd5e1"),
]

for x, y, w, h, label, bg, bd in containers:
    c_rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                   ec=bd, fc=bg, lw=1.2, linestyle="--", zorder=1)
    ax.add_patch(c_rect)
    ax.text(x + w/2, y + h - 0.30, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#64748b", zorder=2)

# -------------------------------------------------------------
# 1. INPUT BLOCKS (Image, Clinical, Audio, Video)
# -------------------------------------------------------------
draw_box(0.6, 6.4, 1.9, 1.15, "Patient Image", "224 x 224 x 3 RGB\n(Posture & Facial Cues)", C_INPUT, B_INPUT)
draw_box(0.6, 4.5, 1.9, 1.15, "Clinical Vitals", "10 Tabular Biomarkers\n(Age, BP, MoCA, Grip, Gait)", C_INPUT, B_INPUT)
draw_box(0.6, 2.6, 1.9, 1.15, "Speech Audio", "WAV / MP3 Waveform\n(Acoustic Speech)", C_INPUT, B_INPUT)
draw_box(0.6, 0.7, 1.9, 1.15, "Gait Video", "16 Frames x 112x112\n(Sequential Motion)", C_INPUT, B_INPUT)

# -------------------------------------------------------------
# 2. FEATURE EXTRACTION BLOCKS
# -------------------------------------------------------------
draw_box(3.3, 6.4, 2.2, 1.15, "Visual CNN Backbone", "GeriatricNet / ResNet50\nOutput: [128-d Vector]", C_BRANCH, B_BRANCH)
draw_box(3.3, 4.5, 2.2, 1.15, "Clinical ANN Branch", "Standard / Attention ANN\nOutput: [32-d Vector]", C_BRANCH, B_BRANCH)
draw_box(3.3, 2.6, 2.2, 1.15, "Audio 2D-CNN Branch", "Librosa Mel-Spectrogram (128x64)\nOutput: [32-d Vector]", C_BRANCH, B_BRANCH)
draw_box(3.3, 0.7, 2.2, 1.15, "Video CNN + BiGRU", "TimeDistributed CNN + BiGRU\nOutput: [64-d Vector]", C_BRANCH, B_BRANCH)

# Arrows from Input -> Extraction
draw_arrow(2.5, 6.97, 3.3, 6.97)
draw_arrow(2.5, 5.07, 3.3, 5.07)
draw_arrow(2.5, 3.17, 3.3, 3.17)
draw_arrow(2.5, 1.27, 3.3, 1.27)

# -------------------------------------------------------------
# 3. FUSION BLOCKS (Explicit 192-d and 256-d Dimensionality)
# -------------------------------------------------------------
draw_box(6.3, 4.6, 2.3, 1.9, "Intermediate Concatenate",
         "layers.Concatenate()\n------------------------\n* 192-d (3 Branches: Image+Clin+Audio)\n* 256-d (4 Branches: with Video)",
         C_FUSION, B_FUSION, title_size=9, sub_size=7.2)

draw_box(6.3, 2.1, 2.3, 1.8, "Deep Dense Fusion",
         "Dense(256) -> BN -> ReLU -> Drop(0.4)\nDense(128) -> BN -> ReLU -> Drop(0.3)\n(Cross-Modal Correlation Learning)",
         C_FUSION, B_FUSION, title_size=9, sub_size=7.2)

# Arrows from Extraction -> Fusion Concatenate
draw_arrow(5.5, 6.97, 6.3, 5.8)
draw_arrow(5.5, 5.07, 6.3, 5.4)
draw_arrow(5.5, 3.17, 6.3, 5.0)
draw_arrow(5.5, 1.27, 6.3, 4.7)

# Arrow from Concatenate -> Deep Dense
draw_arrow(7.45, 4.6, 7.45, 3.9)

# -------------------------------------------------------------
# 4. DECISION & EXPLAINABILITY BLOCKS
# -------------------------------------------------------------
draw_box(9.4, 4.4, 2.3, 3.0, "Softmax Multi-Class\nClassifier Head",
         "P(Y=c|X) Across 5 Syndromes:\n-----------------------------\n1. Frailty\n2. Sarcopenia\n3. Cognitive Impairment\n4. Fall Risk\n5. Depression",
         C_OUTPUT, B_OUTPUT, title_size=9, sub_size=7.8)

draw_box(9.4, 1.1, 2.3, 2.4, "Grad-CAM Explainable AI",
         "Last Conv Layer Gradients (gn_pw)\n-----------------------------\n* Spatial Saliency Localization\n* Thermal JET Heatmap Overlay\n* Clinician Verification Audit",
         C_OUTPUT, B_OUTPUT, title_size=9, sub_size=7.2)

# Arrows from Fusion -> Outputs
draw_arrow(8.6, 3.0, 9.4, 5.6, label="Syndrome Scores")
draw_arrow(5.5, 6.97, 9.4, 2.3, label="CNN Feature Maps", label_y_offset=0.22)

# -------------------------------------------------------------
# 5. DEPLOYMENT BLOCKS
# -------------------------------------------------------------
draw_box(12.5, 4.6, 1.9, 1.7, "FastAPI Backend", "Uvicorn Async ASGI (Port 8000)\nREST API Endpoint: /predict\nPydantic Input Validation\nModel Cache Loader", C_SYSTEM, B_SYSTEM, title_size=8.5, sub_size=7)
draw_box(12.5, 2.1, 1.9, 1.7, "React Vite UI", "Clinical Dashboard (Port 5173)\nInteractive Radar & Bar Visuals\nEvidence-Based Recommendations\nReal-Time Triage (<300ms)", C_SYSTEM, B_SYSTEM, title_size=8.5, sub_size=7)

# Arrows to Deployment
draw_arrow(11.7, 5.9, 12.5, 5.45)
draw_arrow(11.7, 2.3, 12.5, 2.8)
draw_arrow(13.45, 4.6, 13.45, 3.8, label="JSON / Base64", label_y_offset=0.0)

# Diagram Header & Caption
ax.text(7.5, 8.65, "Fig. 1. Overall Block Diagram of GeroFusion Net",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")
ax.text(7.5, 8.35, "Multimodal Input Processing (Image, Clinical, Audio, Video), Feature-Level Fusion (192-d / 256-d), Softmax Classification, and Grad-CAM Explainability",
        ha="center", va="center", fontsize=9.2, color="#475569")

plt.tight_layout()

# Save as PNG
png_path = os.path.join(OUT_DIR, "Figure_Slide_6_Block_Diagram_Entire_Project.png")
fig.savefig(png_path, bbox_inches='tight')
plt.close(fig)

# Save as high-quality JPEG
img = Image.open(png_path).convert("RGB")
jpg_path1 = os.path.join(OUT_DIR, "Figure_Slide_6_Block_Diagram_Entire_Project.jpg")
jpg_path2 = os.path.join(OUT_DIR, "Figure_1_Overall_Block_Diagram_GeroFusion.jpg")
img.save(jpg_path1, "JPEG", quality=95)
img.save(jpg_path2, "JPEG", quality=95)

print(f"Successfully generated:\n- {png_path}\n- {jpg_path1}\n- {jpg_path2}")
