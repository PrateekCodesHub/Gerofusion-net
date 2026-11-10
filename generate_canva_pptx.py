import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

def create_pptx_diagram(output_path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 standard Canva slide
    prs.slide_height = Inches(7.5)
    
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(248, 250, 252) # #f8fafc

    # Color definitions
    C_TEXT_MAIN = RGBColor(15, 23, 42)    # Slate 900
    C_TEXT_MUTED = RGBColor(100, 116, 139) # Slate 500
    C_BORDER_DEFAULT = RGBColor(203, 213, 225)
    
    # Branch 1: Image (Blue)
    BG_BLUE = RGBColor(239, 246, 255)
    BD_BLUE = RGBColor(59, 130, 246)
    
    # Branch 2: Clinical (Emerald Green)
    BG_GREEN = RGBColor(236, 253, 245)
    BD_GREEN = RGBColor(16, 185, 129)
    
    # Branch 3: Audio (Amber/Orange)
    BG_AMBER = RGBColor(254, 243, 199)
    BD_AMBER = RGBColor(245, 158, 11)
    
    # Branch 4: Video (Purple)
    BG_PURPLE = RGBColor(245, 243, 255)
    BD_PURPLE = RGBColor(139, 92, 246)
    
    # Fusion (Indigo/Violet)
    BG_FUSION = RGBColor(238, 242, 255)
    BD_FUSION = RGBColor(99, 102, 241)
    
    # Output / Softmax (Rose/Coral)
    BG_OUTPUT = RGBColor(255, 241, 242)
    BD_OUTPUT = RGBColor(244, 63, 94)

    # Title header
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.733), Inches(0.8))
    tf = header_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "GeroFusion Net: Multimodal Feature-Level Fusion Architecture"
    p.font.name = "Arial"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_MAIN
    
    p2 = tf.add_paragraph()
    p2.text = "Modular Deep Learning Pipeline for Multi-Class Geriatric Syndrome Assessment"
    p2.font.name = "Arial"
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_TEXT_MUTED

    # Helper function to add rounded card
    def add_card(x, y, w, h, bg_color, bd_color, title, subtitle="", badge=""):
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(y), Inches(w), Inches(h)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = bd_color
        shape.line.width = Pt(1.5)
        
        tf = shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.1)
        tf.margin_right = Inches(0.1)
        tf.margin_top = Inches(0.08)
        tf.margin_bottom = Inches(0.08)
        
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_MAIN
        
        if subtitle:
            p_sub = tf.add_paragraph()
            p_sub.alignment = PP_ALIGN.CENTER
            p_sub.text = subtitle
            p_sub.font.name = "Arial"
            p_sub.font.size = Pt(9)
            p_sub.font.color.rgb = RGBColor(51, 65, 85)
            
        if badge:
            p_badge = tf.add_paragraph()
            p_badge.alignment = PP_ALIGN.CENTER
            p_badge.text = badge
            p_badge.font.name = "Arial"
            p_badge.font.size = Pt(9)
            p_badge.font.bold = True
            p_badge.font.color.rgb = bd_color
            
        return shape

    # Helper function for column group banners
    def add_col_header(x, y, w, title):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.35))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = title.upper()
        p.font.name = "Arial"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_MUTED

    # 4 Branch rows
    # y positions for 4 branches
    y_coords = [1.6, 2.85, 4.1, 5.35]
    h_box = 1.05
    
    add_col_header(0.6, 1.25, 2.0, "1. Modality Inputs")
    add_col_header(2.9, 1.25, 2.4, "2. Feature Backbones")
    add_col_header(5.6, 1.25, 3.4, "3. Feature-Level Fusion")
    add_col_header(9.4, 1.25, 3.3, "4. Softmax Classification")

    # Column 1: Inputs
    add_card(0.6, y_coords[0], 2.0, h_box, BG_BLUE, BD_BLUE, "Image Input", "224 x 224 x 3 RGB\n(Posture & Facial Cues)")
    add_card(0.6, y_coords[1], 2.0, h_box, BG_GREEN, BD_GREEN, "Clinical Vitals", "10 Tabular Biomarkers\n(Age, Grip, MoCA, Gait)")
    add_card(0.6, y_coords[2], 2.0, h_box, BG_AMBER, BD_AMBER, "Speech Audio", "Acoustic Audio WAV\n(Pitch, Jitter, Shimmer)")
    add_card(0.6, y_coords[3], 2.0, h_box, BG_PURPLE, BD_PURPLE, "Video Input", "16 Frames x 112 x 112\n(Sequential Gait Kinetics)")

    # Column 2: Backbones & Extracted Features
    add_card(2.9, y_coords[0], 2.4, h_box, BG_BLUE, BD_BLUE, "GeriatricNet CNN", "ResNet50 Backbone\nGlobal Avg Pooling", "Vector: 128-d")
    add_card(2.9, y_coords[1], 2.4, h_box, BG_GREEN, BD_GREEN, "Clinical ANN Branch", "Dense(64) -> Dense(32)\nBatchNorm + Dropout", "Vector: 32-d")
    add_card(2.9, y_coords[2], 2.4, h_box, BG_AMBER, BD_AMBER, "Audio 2D-CNN", "Mel-Spectrogram (128x64)\nConv2D + MaxPool", "Vector: 32-d")
    add_card(2.9, y_coords[3], 2.4, h_box, BG_PURPLE, BD_PURPLE, "Video CNN + BiGRU", "TimeDistributed CNN\nBidirectional GRU", "Vector: 64-d")

    # Column 3: Feature-Level Fusion
    # Fusion Box 1: Concatenation Layer with explicit 192-d vs 256-d callout!
    fusion_cat = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(5.7), Inches(1.6), Inches(3.3), Inches(2.2)
    )
    fusion_cat.fill.solid()
    fusion_cat.fill.fore_color.rgb = BG_FUSION
    fusion_cat.line.color.rgb = BD_FUSION
    fusion_cat.line.width = Pt(2)
    tf_f1 = fusion_cat.text_frame
    tf_f1.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf_f1.word_wrap = True
    
    p = tf_f1.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = "Feature Concatenation Layer"
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_MAIN
    
    p = tf_f1.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    p.text = "layers.Concatenate(axis=-1)"
    p.font.name = "Arial"
    p.font.size = Pt(9.5)
    p.font.color.rgb = RGBColor(71, 85, 105)
    
    p = tf_f1.add_paragraph()
    p.text = "" # blank spacer
    
    # 3-Branch Callout
    p_3b = tf_f1.add_paragraph()
    p_3b.alignment = PP_ALIGN.LEFT
    p_3b.text = "  [3-Branch Fusion]: 192 Dimensions"
    p_3b.font.name = "Arial"
    p_3b.font.size = Pt(10)
    p_3b.font.bold = True
    p_3b.font.color.rgb = RGBColor(2, 132, 199) # Blue
    
    p_3b_sub = tf_f1.add_paragraph()
    p_3b_sub.alignment = PP_ALIGN.LEFT
    p_3b_sub.text = "    Image (128-d) + Clinical (32-d) + Audio (32-d)"
    p_3b_sub.font.name = "Arial"
    p_3b_sub.font.size = Pt(8.5)
    p_3b_sub.font.color.rgb = RGBColor(51, 65, 85)
    
    # 4-Branch Callout
    p_4b = tf_f1.add_paragraph()
    p_4b.alignment = PP_ALIGN.LEFT
    p_4b.text = "  [4-Branch Fusion]: 256 Dimensions"
    p_4b.font.name = "Arial"
    p_4b.font.size = Pt(10)
    p_4b.font.bold = True
    p_4b.font.color.rgb = RGBColor(124, 58, 237) # Purple
    
    p_4b_sub = tf_f1.add_paragraph()
    p_4b_sub.alignment = PP_ALIGN.LEFT
    p_4b_sub.text = "    Image (128-d) + Clinical (32-d) + Audio (32-d) + Video (64-d)"
    p_4b_sub.font.name = "Arial"
    p_4b_sub.font.size = Pt(8.5)
    p_4b_sub.font.color.rgb = RGBColor(51, 65, 85)

    # Fusion Box 2: Deep Cross-Modal Dense Fusion
    fusion_dense = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(5.7), Inches(4.1), Inches(3.3), Inches(2.3)
    )
    fusion_dense.fill.solid()
    fusion_dense.fill.fore_color.rgb = BG_FUSION
    fusion_dense.line.color.rgb = BD_FUSION
    fusion_dense.line.width = Pt(1.5)
    tf_f2 = fusion_dense.text_frame
    tf_f2.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf_f2.word_wrap = True
    
    p = tf_f2.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = "Deep Dense Fusion Network"
    p.font.name = "Arial"
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_MAIN
    
    p_sub = tf_f2.add_paragraph()
    p_sub.alignment = PP_ALIGN.CENTER
    p_sub.text = "Non-Linear Cross-Modal Interaction"
    p_sub.font.name = "Arial"
    p_sub.font.size = Pt(9)
    p_sub.font.color.rgb = RGBColor(71, 85, 105)
    
    items = [
        "• Dense Layer (256 Neurons)",
        "• Batch Normalization + ReLU",
        "• Spatial Dropout (rate = 0.40)",
        "• Dense Layer (128 Neurons)",
        "• Batch Normalization + ReLU + Dropout(0.30)",
        "==> Final Latent Joint Embedding: 128-d"
    ]
    for it in items:
        p_it = tf_f2.add_paragraph()
        p_it.alignment = PP_ALIGN.LEFT
        p_it.text = "  " + it
        p_it.font.name = "Arial"
        p_it.font.size = Pt(8.5)
        p_it.font.bold = ("Final Latent" in it)
        p_it.font.color.rgb = BD_FUSION if ("Final Latent" in it) else RGBColor(51, 65, 85)

    # Column 4: Softmax Classification Head & Outputs
    clf_head = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(9.4), Inches(1.6), Inches(3.3), Inches(4.8)
    )
    clf_head.fill.solid()
    clf_head.fill.fore_color.rgb = BG_OUTPUT
    clf_head.line.color.rgb = BD_OUTPUT
    clf_head.line.width = Pt(2)
    tf_out = clf_head.text_frame
    tf_out.vertical_anchor = MSO_ANCHOR.TOP
    tf_out.word_wrap = True
    tf_out.margin_top = Inches(0.2)
    
    p = tf_out.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = "Softmax Classification Head"
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_MAIN
    
    p_eq = tf_out.add_paragraph()
    p_eq.alignment = PP_ALIGN.CENTER
    p_eq.text = "Dense(5, activation='softmax')"
    p_eq.font.name = "Arial"
    p_eq.font.size = Pt(9.5)
    p_eq.font.bold = True
    p_eq.font.color.rgb = BD_OUTPUT
    
    p_div = tf_out.add_paragraph()
    p_div.alignment = PP_ALIGN.CENTER
    p_div.text = "────────────────────────────"
    p_div.font.size = Pt(8)
    p_div.font.color.rgb = RGBColor(226, 232, 240)
    
    p_lbl = tf_out.add_paragraph()
    p_lbl.alignment = PP_ALIGN.CENTER
    p_lbl.text = "Geriatric Syndrome Probabilities:"
    p_lbl.font.name = "Arial"
    p_lbl.font.size = Pt(10)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = RGBColor(30, 41, 59)
    
    syndromes = [
        ("1. Frailty", "Fried Frailty Phenotype Score"),
        ("2. Sarcopenia", "Muscle Loss / EWGSOP2 Criterion"),
        ("3. Cognitive Impairment", "MoCA Mild/Major Deficits"),
        ("4. Fall Risk", "Tinetti Mobility / Gait Instability"),
        ("5. Depression", "GDS-15 Geriatric Depression Scale")
    ]
    for syn_title, syn_desc in syndromes:
        p_s = tf_out.add_paragraph()
        p_s.text = f"  • {syn_title}"
        p_s.font.name = "Arial"
        p_s.font.size = Pt(9.5)
        p_s.font.bold = True
        p_s.font.color.rgb = RGBColor(159, 18, 57)
        
        p_sd = tf_out.add_paragraph()
        p_sd.text = f"     {syn_desc}"
        p_sd.font.name = "Arial"
        p_sd.font.size = Pt(8)
        p_sd.font.color.rgb = RGBColor(100, 116, 139)

    prs.save(output_path)
    print(f"PowerPoint diagram saved to: {output_path}")

create_pptx_diagram("GeroFusionNet_Canva_Diagram.pptx")
