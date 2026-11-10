from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
import json

SOURCE = "trained_models/GeriatricNet_Standard_metrics.json"
OUTPUT = "NeuroVision_Saved_Training_Result.pptx"
with open(SOURCE, encoding="utf-8") as f:
    result = json.load(f)

prs = Presentation()
prs.slide_width = Inches(16)
prs.slide_height = Inches(9)
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)

NAVY = RGBColor(16, 48, 78)
BLUE = RGBColor(36, 119, 153)
GOLD = RGBColor(238, 164, 50)
PALE = RGBColor(242, 247, 249)
PALE_BLUE = RGBColor(230, 241, 245)
TEXT = RGBColor(52, 68, 82)
MUTED = RGBColor(116, 137, 151)
WHITE = RGBColor(255, 255, 255)


def add_text(x, y, w, h, text, size, color, bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.03)
    tf.margin_right = Inches(0.03)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.01)
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    for run in p.runs:
        run.font.name = "Aptos"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return shape


def rounded(x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line if line else fill
    s.line.width = Pt(1)
    return s


# Header
add_text(0.62, 0.28, 14.8, 0.3, "RESULTS", 11, BLUE, True)
add_text(0.62, 0.72, 14.8, 0.65, "Saved Training Result", 27, NAVY, True)
add_text(0.65, 1.42, 14.4, 0.35, "GeriatricNet + Standard ANN | recorded validation metrics", 13, MUTED)

# Main metric panel
rounded(0.75, 2.15, 5.3, 3.95, NAVY)
add_text(1.12, 2.62, 4.55, 0.35, "VALIDATION ACCURACY", 12, RGBColor(211, 226, 236), True, PP_ALIGN.CENTER)
add_text(1.1, 3.15, 4.6, 1.15, f"{result['accuracy'] * 100:.2f}%", 48, GOLD, True, PP_ALIGN.CENTER)
add_text(1.1, 4.48, 4.6, 0.42, "GeriatricNet + Standard ANN", 17, WHITE, True, PP_ALIGN.CENTER)
add_text(1.18, 5.08, 4.45, 0.45, f"Recorded model size: {result['model_size_mb']:.1f} MB", 12, RGBColor(211, 226, 236), False, PP_ALIGN.CENTER)

# Other saved metrics
cards = [
    (6.55, 2.15, "AUC", f"{result['auc']:.3f}", "Recorded in saved metrics"),
    (10.9, 2.15, "Precision", f"{result['precision'] * 100:.2f}%", "Overall saved metric"),
    (6.55, 4.15, "Recall", f"{result['recall'] * 100:.2f}%", "Overall saved metric"),
    (10.9, 4.15, "Epochs", str(result['epochs_run']), "Training run recorded"),
]
for x, y, label, value, sub in cards:
    rounded(x, y, 3.85, 1.55, PALE, RGBColor(219, 229, 235))
    add_text(x + 0.2, y + 0.18, 3.45, 0.28, label.upper(), 10, BLUE, True, PP_ALIGN.CENTER)
    add_text(x + 0.2, y + 0.48, 3.45, 0.55, value, 23, NAVY, True, PP_ALIGN.CENTER)
    add_text(x + 0.18, y + 1.12, 3.5, 0.22, sub, 9, MUTED, False, PP_ALIGN.CENTER)

# Per-class F1, accurately shown from stored file
rounded(0.75, 6.55, 14.0, 1.55, PALE_BLUE, RGBColor(219, 229, 235))
add_text(1.02, 6.73, 13.4, 0.3, "PER-CLASS F1 SCORE", 10, BLUE, True)
classes = result["per_class"]
slot_w = 2.58
for i, item in enumerate(classes):
    x = 1.02 + i * 2.7
    add_text(x, 7.13, slot_w, 0.27, item["syndrome"], 11, TEXT, True, PP_ALIGN.CENTER)
    add_text(x, 7.48, slot_w, 0.36, f"{item['f1_score'] * 100:.1f}%", 18, NAVY, True, PP_ALIGN.CENTER)

# Footer/source note
add_text(0.62, 8.35, 14.9, 0.25,
         "Source: trained_models/GeriatricNet_Standard_metrics.json | One saved configuration; not a 10-model comparison.",
         9, MUTED)
add_text(15.05, 8.35, 0.45, 0.25, "12/18", 9, MUTED, False, PP_ALIGN.RIGHT)

prs.save(OUTPUT)
print(OUTPUT)
