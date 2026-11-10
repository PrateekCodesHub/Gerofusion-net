from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor


OUT = "NeuroVision_Current_Training_Workflow.pptx"
prs = Presentation()
prs.slide_width = Inches(14)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])
slide.background.fill.solid()
slide.background.fill.fore_color.rgb = RGBColor(248, 250, 252)

NAVY = RGBColor(24, 36, 58)
BLUE = RGBColor(57, 120, 212)
BLUE_BG = RGBColor(226, 237, 255)
GREEN = RGBColor(39, 164, 119)
GREEN_BG = RGBColor(224, 247, 237)
GOLD = RGBColor(223, 155, 25)
GOLD_BG = RGBColor(255, 243, 216)
PURPLE = RGBColor(139, 99, 199)
PURPLE_BG = RGBColor(241, 234, 255)
RED = RGBColor(214, 85, 112)
RED_BG = RGBColor(255, 232, 236)
MUTED = RGBColor(80, 96, 120)
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(91, 111, 136)


def textbox(x, y, w, h, text, size=12, color=NAVY, bold=False,
            align=PP_ALIGN.CENTER, font="Aptos"):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.02)
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_after = Pt(1)
        for run in p.runs:
            run.font.name = font
            run.font.size = Pt(size if i == 0 else max(size - 1, 9))
            run.font.bold = bold if i == 0 else False
            run.font.color.rgb = color if i == 0 else MUTED
    return box


def rounded(x, y, w, h, title, subtitle, fill, edge, title_size=13):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = edge
    sh.line.width = Pt(1.3)
    tf = sh.text_frame
    tf.clear()
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    tf.margin_left = Inches(0.09)
    tf.margin_right = Inches(0.09)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    for i, line in enumerate([title, *subtitle.split("\n")]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = PP_ALIGN.CENTER
        p.space_after = Pt(1)
        for run in p.runs:
            run.font.name = "Aptos"
            run.font.size = Pt(title_size if i == 0 else title_size - 2)
            run.font.bold = i == 0
            run.font.color.rgb = NAVY if i == 0 else MUTED
    return sh


def arrow(x1, y1, x2, y2, color=LINE, width=1.7):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    c.line.end_arrowhead = True
    return c


# Heading
textbox(0.55, 0.23, 12.9, 0.45, "CURRENT MODEL TRAINING WORKFLOW", 11, BLUE, True, PP_ALIGN.LEFT)
textbox(0.55, 0.72, 12.9, 0.55, "Image + Generated Clinical Features", 25, NAVY, True, PP_ALIGN.LEFT)
textbox(0.58, 1.27, 12.6, 0.35,
        "Workflow represented by the current saved training pipeline", 12, MUTED, False, PP_ALIGN.LEFT)

# Branch labels
textbox(0.62, 1.88, 5.7, 0.3, "IMAGE BRANCH", 11, BLUE, True, PP_ALIGN.LEFT)
textbox(0.62, 3.47, 5.7, 0.3, "CLINICAL FEATURE BRANCH", 11, GREEN, True, PP_ALIGN.LEFT)

# Two input branches
rounded(0.65, 2.22, 2.15, 0.82, "Prepared Image", "224 × 224 × 3", BLUE_BG, BLUE)
rounded(3.40, 2.22, 2.55, 0.82, "GeriatricNet CNN", "Image feature extraction", BLUE_BG, BLUE)
rounded(6.55, 2.22, 1.95, 0.82, "Image Features", "128-D vector", BLUE_BG, BLUE)

rounded(0.65, 3.82, 2.15, 0.82, "Generated Features", "10 clinical values", GREEN_BG, GREEN)
rounded(3.40, 3.82, 2.55, 0.82, "Standard ANN", "Dense 64 → 32", GREEN_BG, GREEN)
rounded(6.55, 3.82, 1.95, 0.82, "Clinical Features", "32-D vector", GREEN_BG, GREEN)

# Branch arrows
arrow(2.80, 2.63, 3.40, 2.63, BLUE)
arrow(5.95, 2.63, 6.55, 2.63, BLUE)
arrow(2.80, 4.23, 3.40, 4.23, GREEN)
arrow(5.95, 4.23, 6.55, 4.23, GREEN)

# Fusion and classification
rounded(9.20, 2.90, 1.75, 0.98, "Concatenate", "128 + 32 = 160-D", GOLD_BG, GOLD)
rounded(11.45, 2.90, 1.85, 0.98, "Dense Layers", "256 → 128", PURPLE_BG, PURPLE)
arrow(8.50, 2.63, 8.86, 2.63, BLUE)
arrow(8.50, 4.23, 8.86, 4.23, GREEN)
arrow(8.86, 2.63, 8.86, 3.12, GOLD)
arrow(8.86, 4.23, 8.86, 3.67, GOLD)
arrow(8.86, 3.39, 9.20, 3.39, GOLD)
arrow(10.95, 3.39, 11.45, 3.39, PURPLE)

rounded(11.45, 4.45, 1.85, 0.85, "Softmax Output", "5 classes", RED_BG, RED)
arrow(12.38, 3.88, 12.38, 4.45, RED)

# Output classes
textbox(9.15, 5.52, 4.25, 0.34, "FIVE PREDICTED CATEGORIES", 10, MUTED, True)
labels = ["Frailty", "Sarcopenia", "Cognitive\nImpairment", "Fall Risk", "Depression"]
start_x, gap, bw = 8.95, 0.12, 0.82
for i, label in enumerate(labels):
    x = start_x + i * (bw + gap)
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(5.92), Inches(bw), Inches(0.7))
    sh.fill.solid(); sh.fill.fore_color.rgb = NAVY
    sh.line.color.rgb = NAVY
    tf = sh.text_frame; tf.clear(); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.03); tf.margin_right = Inches(0.03)
    p = tf.paragraphs[0]; p.text = label; p.alignment = PP_ALIGN.CENTER
    for run in p.runs:
        run.font.name = "Aptos"; run.font.size = Pt(9); run.font.bold = True; run.font.color.rgb = WHITE

# Honest scope note and footer
note = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(5.15), Inches(7.65), Inches(1.2))
note.fill.solid(); note.fill.fore_color.rgb = WHITE
note.line.color.rgb = RGBColor(215, 224, 232)
tf = note.text_frame; tf.clear(); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
tf.margin_left = Inches(0.2); tf.margin_right = Inches(0.18)
p = tf.paragraphs[0]; p.text = "Training data note"; p.space_after = Pt(5)
for r in p.runs: r.font.name="Aptos"; r.font.size=Pt(12); r.font.bold=True; r.font.color.rgb=NAVY
p = tf.add_paragraph(); p.text = "Images come from the prepared image folders. Clinical values are generated by the training code; CSV and audio files are not loaded in this training run."
p.alignment = PP_ALIGN.LEFT
for r in p.runs: r.font.name="Aptos"; r.font.size=Pt(10); r.font.color.rgb=MUTED

textbox(0.55, 7.1, 12.9, 0.2, "NeuroVision | Current training pipeline", 9, MUTED, False, PP_ALIGN.LEFT)

prs.save(OUT)
print(OUT)
