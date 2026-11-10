"""Regenerate slides 14–18 in the revised NeuroVision review deck.

Slides 1–13 are preserved by copying their existing editable shapes into a
fresh deck. The final five slides are recreated with metrics and claims drawn
from the project's saved run and code.
"""
from copy import deepcopy
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

SOURCE = "NeuroVision_v3_Project_Review.pptx (1).pptx"
OUTPUT = "NeuroVision_v3_Project_Review_Updated_Fixed.pptx"
METRICS = "trained_models/GeriatricNet_Standard_metrics.json"

with open(METRICS, encoding="utf-8") as f:
    metrics = json.load(f)
src = Presentation(SOURCE)
prs = src
target_slides = list(prs.slides)[13:18]
source_references = [sh.text.strip() for sh in src.slides[16].shapes
                     if sh.has_text_frame and sh.text.strip()]


def clear_slide(slide):
    """Remove slide-local shapes while preserving group properties and rels."""
    sp_tree = slide.shapes._spTree
    for element in list(sp_tree):
        if element.tag.endswith("}spTree"):
            continue
        if element.tag.endswith("}nvGrpSpPr") or element.tag.endswith("}grpSpPr"):
            continue
        sp_tree.remove(element)


# Reuse the original slides and their layouts/master/media relationships.
# This keeps the first thirteen slides fully intact and avoids broken images.
for slide in target_slides:
    clear_slide(slide)

# restrained Canva-like palette, consistent with the existing deck
NAVY = RGBColor(11, 46, 79)
BLUE = RGBColor(28, 114, 147)
TEAL = RGBColor(50, 145, 161)
GOLD = RGBColor(235, 164, 57)
INK = RGBColor(40, 57, 72)
MUTED = RGBColor(104, 124, 140)
PAPER = RGBColor(249, 250, 248)
PALE = RGBColor(237, 244, 245)
PALE2 = RGBColor(246, 247, 244)
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(216, 226, 230)
GREEN = RGBColor(73, 142, 112)
ORANGE_PALE = RGBColor(255, 244, 224)


def clone_shape(shape, target):
    """Copy editable OOXML shape into target slide; preserve original content."""
    el = deepcopy(shape.element)
    target.shapes._spTree.insert_element_before(el, "p:extLst")


def add_text(slide, x, y, w, h, text, size=16, color=INK, bold=False,
             font="Aptos", align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE,
             margin=0.04):
    sh = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_after = Pt(1)
        for r in p.runs:
            r.font.name = font
            r.font.size = Pt(size if i == 0 else size - 1)
            r.font.bold = bold if i == 0 else False
            r.font.color.rgb = color if i == 0 else MUTED
    return sh


def rounded(slide, x, y, w, h, fill=WHITE, line=LINE, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line; sh.line.width = Pt(1)
    return sh


def card(slide, x, y, w, h, title, body, accent=BLUE, fill=WHITE,
         title_size=17, body_size=13):
    rounded(slide, x, y, w, h, fill, LINE)
    # short color tab gives the card a handmade/editorial feel
    tab = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y+0.12), Inches(0.07), Inches(h-0.24))
    tab.fill.solid(); tab.fill.fore_color.rgb = accent; tab.line.fill.background()
    add_text(slide, x+0.23, y+0.14, w-0.42, 0.36, title, title_size, NAVY, True)
    add_text(slide, x+0.23, y+0.58, w-0.43, h-0.7, body, body_size, INK, False,
             valign=MSO_ANCHOR.TOP, margin=0.02)


def heading(slide, kicker, title, subtitle=None, page=None):
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = PAPER
    add_text(slide, 0.68, 0.28, 10.0, 0.3, kicker.upper(), 10, BLUE, True)
    add_text(slide, 0.68, 0.67, 18.5, 0.65, title, 29, NAVY, True, font="Georgia")
    if subtitle:
        add_text(slide, 0.7, 1.38, 18.0, 0.35, subtitle, 13, MUTED)
    if page is not None:
        add_text(slide, 18.3, 10.82, 1.0, 0.2, f"{page} / 18", 9, MUTED, align=PP_ALIGN.RIGHT)


def line_arrow(slide, x1, y1, x2, y2, color=BLUE, width=2):
    sh = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    sh.line.color.rgb = color; sh.line.width = Pt(width); sh.line.end_arrowhead = True
    return sh


# Slide 14: actual saved result, explicitly one configuration.
s = target_slides[0]
heading(s, "Results", "The saved training run", "One recorded configuration, reported with its validation metrics.", 14)
rounded(s, 0.75, 2.05, 6.5, 4.0, NAVY, NAVY)
add_text(s, 1.05, 2.42, 5.9, 0.35, "GERIATRICNET  +  STANDARD ANN", 13, RGBColor(210, 228, 237), True, align=PP_ALIGN.CENTER)
add_text(s, 1.0, 3.03, 6.0, 1.12, f"{metrics['accuracy']*100:.2f}%", 52, GOLD, True, font="Georgia", align=PP_ALIGN.CENTER)
add_text(s, 1.05, 4.25, 5.9, 0.35, "validation accuracy", 16, WHITE, True, align=PP_ALIGN.CENTER)
add_text(s, 1.05, 5.18, 5.9, 0.35, f"Saved model size: {metrics['model_size_mb']:.1f} MB  ·  {metrics['epochs_run']} epochs", 12, RGBColor(210, 228, 237), align=PP_ALIGN.CENTER)

for x, y, title, val, sub in [
    (7.7, 2.05, "AUC", f"{metrics['auc']:.3f}", "recorded in metrics file"),
    (13.15, 2.05, "Precision", f"{metrics['precision']*100:.2f}%", "overall recorded value"),
    (7.7, 4.05, "Recall", f"{metrics['recall']*100:.2f}%", "overall recorded value"),
    (13.15, 4.05, "Validation split", "20%", "random 80 / 20 split"),
]:
    rounded(s, x, y, 4.85, 1.6, WHITE, LINE)
    add_text(s, x+0.18, y+0.16, 4.49, 0.3, title.upper(), 10, BLUE, True, align=PP_ALIGN.CENTER)
    add_text(s, x+0.18, y+0.48, 4.49, 0.58, val, 27, NAVY, True, font="Georgia", align=PP_ALIGN.CENTER)
    add_text(s, x+0.18, y+1.16, 4.49, 0.22, sub, 10, MUTED, align=PP_ALIGN.CENTER)

# Per-class F1 strip
rounded(s, 0.75, 6.55, 17.25, 1.48, PALE, LINE)
add_text(s, 1.0, 6.68, 3.0, 0.28, "PER-CLASS F1 SCORE", 10, BLUE, True)
for i, item in enumerate(metrics["per_class"]):
    x = 1.1 + i*3.32
    add_text(s, x, 7.08, 2.9, 0.28, item["syndrome"], 12, INK, True, align=PP_ALIGN.CENTER)
    add_text(s, x, 7.42, 2.9, 0.35, f"{item['f1_score']*100:.1f}%", 20, NAVY, True, font="Georgia", align=PP_ALIGN.CENTER)
add_text(s, 0.78, 8.36, 17.3, 0.52,
         "This is the saved result for one configuration. Other accuracy figures shown elsewhere in the app are not part of this metrics file.",
         11, MUTED)


# Slide 15: transparent limitations grounded in artifacts and training code.
s = target_slides[1]
heading(s, "What the current run tells us", "What the prototype can—and cannot—show", "A short interpretation of the current implementation and saved result.", 15)
card(s, 0.8, 2.05, 8.3, 2.05, "What is implemented",
     "The saved training pipeline uses prepared images and generated clinical features. The model predicts one of five project categories.",
     accent=TEAL, fill=WHITE, title_size=18, body_size=14)
card(s, 9.55, 2.05, 8.8, 2.05, "What this result means",
     "68.67% is the validation accuracy recorded for GeriatricNet + Standard ANN in the saved metrics file.",
     accent=GOLD, fill=WHITE, title_size=18, body_size=14)
card(s, 0.8, 4.5, 8.3, 2.05, "Current data limitation",
     "Several image labels use facial-expression images as proxies for health-risk classes. Clinical feature values are generated in code, not read from the CSV files for this run.",
     accent=GOLD, fill=PALE2, title_size=18, body_size=14)
card(s, 9.55, 4.5, 8.8, 2.05, "Next steps",
     "Train and evaluate with verified patient-level data, check each class carefully, and report results from saved evaluation files. Treat outputs as research estimates, not diagnoses.",
     accent=GREEN, fill=PALE2, title_size=18, body_size=14)
rounded(s, 0.8, 7.05, 17.55, 0.72, ORANGE_PALE, RGBColor(239, 218, 180))
add_text(s, 1.05, 7.16, 17.0, 0.46,
         "The current score is a starting point for a prototype; it needs stronger data and validation before real-world use.",
         13, NAVY, True, align=PP_ALIGN.CENTER)


# Slide 16: conclusion / spoken close.
s = target_slides[2]
heading(s, "Conclusion", "A prototype built to explore risk assessment", "The main contribution is an end-to-end project that connects model development with an interactive application.", 16)
# Large pull quote, deliberately sparse rather than a dense template.
rounded(s, 0.85, 2.0, 17.4, 2.25, NAVY, NAVY)
add_text(s, 1.35, 2.38, 16.4, 1.35,
         "NeuroVision brings together image analysis, a clinical-feature branch, and a simple interface for exploring five health-risk categories.",
         23, WHITE, False, font="Georgia", align=PP_ALIGN.CENTER)
card(s, 0.85, 4.85, 5.35, 1.65, "Project contribution",
     "A working research prototype with a CNN–ANN model path and web interface.", accent=TEAL, fill=WHITE, title_size=16, body_size=13)
card(s, 6.65, 4.85, 5.35, 1.65, "Current evidence",
     "One saved GeriatricNet + Standard ANN run, with 68.67% validation accuracy.", accent=GOLD, fill=WHITE, title_size=16, body_size=13)
card(s, 12.45, 4.85, 5.8, 1.65, "Next milestone",
     "Use verified clinical data and validate results with a suitable held-out dataset.", accent=GREEN, fill=WHITE, title_size=16, body_size=13)
add_text(s, 0.9, 7.15, 17.2, 0.58,
         "Thank you  ·  Questions?", 20, NAVY, True, font="Georgia", align=PP_ALIGN.CENTER)


# Slide 17: retain existing references content, restyled lightly and kept editable.
s = target_slides[3]
heading(s, "References", "Sources behind the methods", "Selected papers related to deep learning, medical AI, and explainability.", 17)
for i, ref in enumerate(source_references):
    col = 0 if i < 4 else 1
    row = i if col == 0 else i-4
    x = 0.85 + col*9.0
    y = 2.05 + row*1.9
    card(s, x, y, 8.35, 1.55, f"{i+1:02d}  ·  Reference", ref,
         accent=[TEAL, BLUE, GOLD, GREEN][i % 4], fill=WHITE, title_size=12, body_size=10)
add_text(s, 0.9, 10.45, 17.7, 0.25,
         "References provide background on methods; they are not evidence of this project's measured performance.",
         10, MUTED)


# Slide 18: short Q&A buffer for the live review.
s = target_slides[4]
s.background.fill.solid(); s.background.fill.fore_color.rgb = NAVY
add_text(s, 1.2, 2.1, 17.6, 0.45, "NEUROVISION  ·  PROJECT REVIEW", 12, RGBColor(168, 207, 218), True, align=PP_ALIGN.CENTER)
add_text(s, 1.2, 3.15, 17.6, 1.1, "Questions & discussion", 38, WHITE, True, font="Georgia", align=PP_ALIGN.CENTER)
add_text(s, 3.4, 4.62, 13.2, 0.62,
         "Thank you for your time", 19, GOLD, False, font="Georgia", align=PP_ALIGN.CENTER)
add_text(s, 1.2, 10.82, 17.6, 0.22, "ECE(AI) Honors Project Review", 10, RGBColor(168, 207, 218), align=PP_ALIGN.CENTER)

prs.save(OUTPUT)
print(OUTPUT)
