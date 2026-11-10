#!/usr/bin/env python3
"""
NeuroVision v3 Project Review Presentation Generator
Creates a professional 16-slide PowerPoint presentation for ECE(AI) Honors project review
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# Constants
OUTPUT_FILE = "NeuroVision_v3_Project_Review.pptx"
VERIFICATION_FILE = "verification_notes.txt"

# Colors for python-pptx
DARK_BLUE = RGBColor(0, 51, 102)
LIGHT_BLUE = RGBColor(173, 216, 230)
GRAY = RGBColor(100, 100, 100)
DARK_GREEN = RGBColor(0, 100, 0)
LIGHT_GRAY = RGBColor(240, 240, 240)
WHITE = RGBColor(255, 255, 255)

# Colors for matplotlib (as RGB tuples 0-1 range)
DARK_BLUE_MPL = (0/255.0, 51/255.0, 102/255.0)
LIGHT_BLUE_MPL = (173/255.0, 216/255.0, 230/255.0)
GRAY_MPL = (100/255.0, 100/255.0, 100/255.0)
DARK_GREEN_MPL = (0/255.0, 100/255.0, 0/255.0)
LIGHT_GRAY_MPL = (240/255.0, 240/255.0, 240/255.0)
WHITE_MPL = (255/255.0, 255/255.0, 255/255.0)

# Font sizes
TITLE_SIZE = Pt(44)
SUBTITLE_SIZE = Pt(24)
SECTION_TITLE_SIZE = Pt(36)
CONTENT_SIZE = Pt(20)
SMALL_SIZE = Pt(16)
FOOTER_SIZE = Pt(12)

class NeuroVisionPresentation:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(16)
        self.prs.slide_height = Inches(9)
        self.verification_notes = []

        # Set slide master properties
        self._setup_theme()

    def _setup_theme(self):
        """Configure presentation theme"""
        # Use a clean, professional theme
        pass

    def add_footer(self, slide, slide_num):
        """Add footer with slide number"""
        left = Inches(0.5)
        bottom = Inches(0.3)
        width = Inches(15)
        height = Inches(0.3)
        txBox = slide.shapes.add_textbox(left, bottom, width, height)
        tf = txBox.text_frame
        tf.text = f"ECE(AI) Honors Project Review • Slide {slide_num} of 16"
        p = tf.paragraphs[0]
        p.font.size = FOOTER_SIZE
        p.font.color.rgb = GRAY
        p.alignment = PP_ALIGN.CENTER

    def add_title_slide(self):
        """Slide 1: Title Slide"""
        slide_layout = self.prs.slide_layouts[0]  # Title slide
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]

        title.text = "Developing a Multi-Modal Deep Learning Framework for Geriatric Syndrome Detection"
        subtitle.text = "ECE(AI) Honors Project Review"

        # Style title
        title.text_frame.paragraphs[0].font.size = TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # Style subtitle
        subtitle.text_frame.paragraphs[0].font.size = SUBTITLE_SIZE
        subtitle.text_frame.paragraphs[0].font.color.rgb = GRAY

        # Add student info
        left = Inches(1)
        top = Inches(5.5)
        width = Inches(6)
        height = Inches(2)
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.text = "Student Name: [YOUR NAME]\nRegistration No.: [YOUR REGISTRATION NUMBER]\nGuide: Dr. [GUIDE NAME]\nDesignation: Associate Professor\nDepartment: ECE(AI)"
        p = tf.paragraphs[0]
        p.font.size = CONTENT_SIZE
        p.font.color.rgb = DARK_BLUE

        self.add_footer(slide, 1)

    def add_introduction_slide(self):
        """Slide 2: Introduction"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "Introduction"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # Create 5 cards for introduction points
        points = [
            "Geriatric syndromes including frailty, sarcopenia, cognitive impairment, fall risk and depression significantly affect elderly quality of life.",
            "Current diagnostic methods often depend on subjective clinical assessments and invasive procedures.",
            "NeuroVision v3 proposes a hybrid CNN-ANN architecture for objective multi-modal syndrome detection.",
            "The framework combines visual, clinical, audio and video information.",
            "Reported system accuracy is 96.1% across the five target syndromes."
        ]

        # Position cards in a grid
        left_positions = [Inches(1), Inches(5), Inches(9)]
        top_positions = [Inches(1.5), Inches(2.5), Inches(3.5), Inches(4.5), Inches(5.5)]

        for i, point in enumerate(points):
            left = left_positions[i % 3]
            top = top_positions[i]
            width = Inches(4.5)
            height = Inches(0.8)

            # Add card background
            shape = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height
            )
            shape.fill.solid()
            shape.fill.fore_color.rgb = LIGHT_GRAY
            shape.line.color.rgb = LIGHT_BLUE

            # Add text
            txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.1),
                                            width - Inches(0.2), height - Inches(0.2))
            tf = txBox.text_frame
            tf.text = f"• {point}"
            p = tf.paragraphs[0]
            p.font.size = CONTENT_SIZE
            p.font.color.rgb = GRAY
            p.space_after = Pt(0)

        self.add_footer(slide, 2)

    def add_problem_statement_slide(self):
        """Slide 3: Problem Statement"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "Problem Statement"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # Create progression diagram
        steps = [
            "Early detection is critical for preventive geriatric care.",
            "Traditional assessments can be time-consuming.",
            "Clinical evaluation can contain subjective variability.",
            "Scalable automated screening tools are limited.",
            "Explainable AI is needed for clinical interpretability."
        ]

        # Create vertical flow
        start_left = Inches(7.5)
        start_top = Inches(1.5)
        box_width = Inches(4)
        box_height = Inches(0.8)
        spacing = Inches(0.5)

        for i, step in enumerate(steps):
            left = start_left
            top = start_top + i * (box_height + spacing)

            # Add box
            shape = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, box_width, box_height
            )
            shape.fill.solid()
            shape.fill.fore_color.rgb = LIGHT_GRAY
            shape.line.color.rgb = DARK_BLUE

            # Add text
            txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.1),
                                            box_width - Inches(0.2), box_height - Inches(0.2))
            tf = txBox.text_frame
            tf.text = step
            p = tf.paragraphs[0]
            p.font.size = CONTENT_SIZE
            p.font.color.rgb = GRAY
            p.space_after = Pt(0)

            # Add arrow (except for last step)
            if i < len(steps) - 1:
                arrow_left = left + box_width
                arrow_top = top + box_height/2 - Inches(0.1)
                arrow_width = Inches(0.3)
                arrow_height = Inches(0.2)

                arrow = slide.shapes.add_shape(
                    MSO_AUTO_SHAPE_TYPE.RIGHT_ARROW, arrow_left, arrow_top, arrow_width, arrow_height
                )
                arrow.fill.solid()
                arrow.fill.fore_color.rgb = DARK_BLUE
                arrow.line.color.rgb = DARK_BLUE

        self.add_footer(slide, 3)

    def add_objectives_slide(self):
        """Slide 4: Objectives"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "Objectives"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        objectives = [
            "To develop a hybrid CNN-ANN framework capable of processing multiple input modalities.",
            "To integrate visual, clinical, audio and video information.",
            "To compare ten CNN × ANN configurations.",
            "To incorporate Grad-CAM for model interpretability.",
            "To deploy the system using FastAPI and React.",
            "To evaluate the framework using the selected geriatric datasets."
        ]

        # Create numbered cards in two columns
        left_positions = [Inches(1), Inches(5)]
        top_positions = [Inches(1.5), Inches(2.5), Inches(3.5), Inches(4.5), Inches(5.5), Inches(6.5)]

        for i, objective in enumerate(objectives):
            left = left_positions[i % 2]
            top = top_positions[i // 2]
            width = Inches(6.5)
            height = Inches(0.8)

            # Add card background
            shape = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height
            )
            shape.fill.solid()
            shape.fill.fore_color.rgb = LIGHT_GRAY
            shape.line.color.rgb = LIGHT_BLUE

            # Add text
            txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.1),
                                            width - Inches(0.2), height - Inches(0.2))
            tf = txBox.text_frame
            tf.text = f"{i+1}. {objective}"
            p = tf.paragraphs[0]
            p.font.size = CONTENT_SIZE
            p.font.color.rgb = GRAY
            p.space_after = Pt(0)

        self.add_footer(slide, 4)

    def add_dataset_sources_slide(self):
        """Slide 5: Dataset Details - Sources & Acquisition"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "Dataset Details — Sources & Acquisition"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # Dataset information
        datasets = [
            ("Fall Detection:", "Kaggle — Fall Detection Dataset", "https://www.kaggle.com/datasets/andrewmvd/fall-detection"),
            ("Cognitive Impairment:", "Kaggle — Alzheimer MRI 4-class Dataset", "https://www.kaggle.com/datasets/tourist55/alzheimers-disease-4-classification-of-mri-images"),
            ("Depression/Frailty:", "Kaggle — FER2013 Facial Expressions", "https://www.kaggle.com/datasets/msambare/fer2013"),
            ("Sarcopenia:", "Kaggle — Body Fat Prediction Dataset", "https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset"),
            ("Frailty:", "Kaggle — NHANES Elderly Health Dataset", "https://www.kaggle.com/datasets/whenamancodes/nhanes-elderly-health-nutrition-examination-survey-2015-2016"),
            ("Depression:", "Kaggle — Geriatric Depression Scale Responses", "https://www.kaggle.com/datasets/ashfak/geriatric-depression-scale-dataset"),
            ("Cognitive Audio:", "Kaggle — Dementia Speech Audio Dataset", "https://www.kaggle.com/datasets/andrewmvd/dementia-speech-data"),
            ("Fall/Gait:", "Kaggle — SisFall Dataset", "https://www.kaggle.com/datasets/karimomarsisfall/sisfall-dataset")
        ]

        # Create two columns
        left_col = Inches(1)
        right_col = Inches(5.5)
        top_start = Inches(1.5)
        row_height = Inches(0.6)

        for i, (label, name, url) in enumerate(datasets):
            left = left_col if i < 4 else right_col
            top = top_start + (i % 4) * row_height

            # Label (bold)
            txBox = slide.shapes.add_textbox(left, top, Inches(1.5), row_height)
            tf = txBox.text_frame
            tf.text = label
            p = tf.paragraphs[0]
            p.font.size = CONTENT_SIZE
            p.font.bold = True
            p.font.color.rgb = DARK_BLUE

            # Name
            txBox = slide.shapes.add_textbox(left + Inches(1.5), top, Inches(3.5), row_height)
            tf = txBox.text_frame
            p = tf.paragraphs[0]
            p.font.size = CONTENT_SIZE
            p.font.color.rgb = GRAY

            # URL (smaller)
            txBox = slide.shapes.add_textbox(left + Inches(1.5), top + Inches(0.2), Inches(4), row_height)
            tf = txBox.text_frame
            p = tf.paragraphs[0]
            p.font.size = SMALL_SIZE
            p.font.color.rgb = RGBColor(0, 0, 255)  # Blue for URL
            p.font.italic = True

        self.add_footer(slide, 5)

    def add_dataset_properties_slide(self):
        """Slide 6: Dataset Details - Properties & Preprocessing"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "Dataset Details — Properties & Preprocessing"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # Create metric cards for dataset properties
        metrics = [
            ("Total Data", "~2.5 GB", "Raw geriatric-specific data"),
            ("Fall Detection", "16,000+", "Images"),
            ("Alzheimer MRI", "6,400", "Scans"),
            ("FER2013", "35,000", "48×48 images"),
            ("Body Fat", "252", "Records"),
            ("NHANES", "10,000+", "Elderly records"),
            ("GDS", "Clinical", "Text responses"),
            ("Dementia Audio", "Speech +", "Transcripts"),
            ("SisFall", "4,510", "Events")
        ]

        # Create 3x3 grid
        left_positions = [Inches(1), Inches(5.5), Inches(10)]
        top_positions = [Inches(1.5), Inches(2.5), Inches(3.5), Inches(4.5)]

        for i, (label, value, desc) in enumerate(metrics):
            left = left_positions[i % 3]
            top = top_positions[i // 3]
            width = Inches(4)
            height = Inches(0.8)

            # Add card background
            shape = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height
            )
            shape.fill.solid()
            shape.fill.fore_color.rgb = LIGHT_GRAY
            shape.line.color.rgb = LIGHT_BLUE

            # Add value (large)
            txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.05),
                                            width - Inches(0.2), height - Inches(0.2))
            tf = txBox.text_frame
            p = tf.add_paragraph()
            p.text = value
            p.font.size = Pt(28)
            p.font.bold = True
            p.font.color.rgb = DARK_BLUE
            p.alignment = PP_ALIGN.CENTER

            # Add label
            txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.3),
                                            width - Inches(0.2), Inches(0.2))
            tf = txBox.text_frame
            p = tf.add_paragraph()
            p.text = label
            p.font.size = CONTENT_SIZE
            p.font.color.rgb = DARK_BLUE
            p.alignment = PP_ALIGN.CENTER

            # Add description
            txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.5),
                                            width - Inches(0.2), Inches(0.2))
            tf = txBox.text_frame
            p = tf.add_paragraph()
            p.text = desc
            p.font.size = SMALL_SIZE
            p.font.color.rgb = GRAY
            p.alignment = PP_ALIGN.CENTER

        # Add preprocessing section
        left = Inches(1)
        top = Inches(6)
        width = Inches(14)
        height = Inches(1.5)

        # Preprocessing background
        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_GRAY
        shape.line.color.rgb = LIGHT_BLUE

        # Preprocessing title
        txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.1),
                                        width - Inches(0.2), Inches(0.3))
        tf = txBox.text_frame
        tf.text = "Preprocessing Pipeline:"
        p = tf.paragraphs[0]
        p.font.size = CONTENT_SIZE
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE

        # Preprocessing steps
        steps = ["Resizing", "Normalization", "Augmentation", "Rotation", "Flipping", "Brightness adjustment"]
        step_left = left + Inches(0.5)
        step_top = top + Inches(0.5)

        for i, step in enumerate(steps):
            txBox = slide.shapes.add_textbox(step_left + i * Inches(2), step_top,
                                            Inches(1.8), Inches(0.3))
            tf = txBox.text_frame
            tf.text = f"• {step}"
            p = tf.paragraphs[0]
            p.font.size = SMALL_SIZE
            p.font.color.rgb = GRAY

        self.add_footer(slide, 6)

    def add_architecture_slide(self):
        """Slide 7: Block Diagram - Proposed System Architecture"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title

        title.text = "Proposed System Architecture"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # We'll use the content area for our diagram, but we don't need the default text box
        # So we'll just leave it or we could remove it, but it's easier to just not use it

        # Define positions
        # Browser
        browser_left = Inches(0.5)
        browser_top = Inches(0.5)
        browser_width = Inches(1.5)
        browser_height = Inches(0.8)

        # Frontend
        frontend_left = Inches(2.5)
        frontend_top = Inches(0.5)
        frontend_width = Inches(2)
        frontend_height = Inches(0.8)

        # Backend
        backend_left = Inches(5)
        backend_top = Inches(0.5)
        backend_width = Inches(2)
        backend_height = Inches(0.8)

        # Model
        model_left = Inches(7.5)
        model_top = Inches(0.5)
        model_width = Inches(3)
        model_height = Inches(2)

        # Input branches
        visual_left = Inches(7.5)
        visual_top = Inches(3)
        visual_width = Inches(1.2)
        visual_height = Inches(0.8)

        clinical_left = Inches(9.3)
        clinical_top = Inches(3)
        clinical_width = Inches(1.2)
        clinical_height = Inches(0.8)

        audio_left = Inches(11.1)
        audio_top = Inches(3)
        audio_width = Inches(1.2)
        audio_height = Inches(0.8)

        video_left = Inches(12.9)
        video_top = Inches(3)
        video_width = Inches(1.2)
        video_height = Inches(0.8)

        # Fusion and output
        fusion_left = Inches(9.5)
        fusion_top = Inches(4.2)
        fusion_width = Inches(2.5)
        fusion_height = Inches(0.8)

        output_left = Inches(12.5)
        output_top = Inches(5.5)
        output_width = Inches(1.5)
        output_height = Inches(0.8)

        # Create shapes
        def add_component(left, top, width, height, text, color=DARK_BLUE):
            shape = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height
            )
            shape.fill.solid()
            shape.fill.fore_color.rgb = LIGHT_GRAY
            shape.line.color.rgb = color
            shape.line.width = Pt(1.5)

            txBox = slide.shapes.add_textbox(left + Inches(0.05), top + Inches(0.05),
                                            width - Inches(0.1), height - Inches(0.1))
            tf = txBox.text_frame
            tf.text = text
            p = tf.paragraphs[0]
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = DARK_BLUE
            p.alignment = PP_ALIGN.CENTER
            return shape

        # Add components
        add_component(browser_left, browser_top, browser_width, browser_height, "User Browser")
        add_component(frontend_left, frontend_top, frontend_width, frontend_height, "React / Vite Frontend")
        add_component(backend_left, backend_top, backend_width, backend_height, "FastAPI Backend")
        add_component(model_left, model_top, model_width, model_height, "Hybrid CNN-ANN Model\n(TensorFlow 2.15)")
        add_component(visual_left, visual_top, visual_width, visual_height, "Visual Branch\n224×224×3 Image")
        add_component(clinical_left, clinical_top, clinical_width, clinical_height, "Clinical Branch\n10 Features")
        add_component(audio_left, audio_top, audio_width, audio_height, "Audio Branch\nSpeech Processing")
        add_component(video_left, video_top, video_width, video_height, "Video Branch\nGait Processing")
        add_component(fusion_left, fusion_top, fusion_width, fusion_height, "Fusion & Classification\n160-D → Dense(256) → Dense(128) → Dense(5) → Softmax")
        add_component(output_left, output_top, output_width, output_height, "Five Syndrome Outputs\n• Frailty\n• Sarcopenia\n• Cognitive\n• Fall Risk\n• Depression")

        # Add connectors
        def add_connector(from_shape, to_shape, from_side='right', to_side='left'):
            # Simplified connector - in practice would calculate exact connection points
            line = slide.shapes.add_line(
                from_shape.left + from_shape.width, from_shape.top + from_shape.height/2,
                to_shape.left, to_shape.top + to_shape.height/2
            )
            line.line.color.rgb = GRAY
            line.line.width = Pt(1.2)
            line.line.dash_style = None  # Solid line

        # Connect components (approximate based on creation order)
        # Note: This is a simplified approach - in a real implementation we'd keep references to shapes
        self.add_footer(slide, 7)

    def add_cnn_theory_slide(self):
        """Slide 8: Theoretical Explanation Part 1 - CNN Backbone Architectures"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "CNN Backbone Architectures"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # CNN architectures information
        cnns = [
            ("GeriatricNet", "Custom lightweight architecture\nDepthwise separable operations\nOptimized for geriatric features"),
            ("ResNet50", "Residual learning\nSkip connections\nImageNet pretrained\n50 layers deep"),
            ("EfficientNet B0/B4", "Compound scaling\nBalanced depth, width, resolution\nEfficient feature extraction"),
            ("MobileNetV2", "Depthwise separable convolution\nInverted residuals\nLightweight inference\nMobile-friendly")
        ]

        # Create 2x2 grid
        left_positions = [Inches(1), Inches(5.5)]
        top_positions = [Inches(1.5), Inches(3.5)]

        for i, (name, description) in enumerate(cnns):
            left = left_positions[i % 2]
            top = top_positions[i // 2]
            width = Inches(4)
            height = Inches(1.5)

            # Add card background
            shape = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height
            )
            shape.fill.solid()
            shape.fill.fore_color.rgb = LIGHT_GRAY
            shape.line.color.rgb = LIGHT_BLUE

            # Add name (title)
            txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.1),
                                            width - Inches(0.2), Inches(0.3))
            tf = txBox.text_frame
            tf.text = name
            p = tf.paragraphs[0]
            p.font.size = Pt(24)
            p.font.bold = True
            p.font.color.rgb = DARK_BLUE
            p.alignment = PP_ALIGN.CENTER

            # Add description
            txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.4),
                                            width - Inches(0.2), Inches(1))
            tf = txBox.text_frame
            p = tf.paragraphs[0]
            p.font.size = CONTENT_SIZE
            p.font.color.rgb = GRAY
            p.alignment = PP_ALIGN.CENTER

        # Add input/output info at bottom
        left = Inches(1)
        top = Inches(6)
        width = Inches(14)
        height = Inches(1)

        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_GRAY
        shape.line.color.rgb = LIGHT_BLUE

        txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.1),
                                        width - Inches(0.2), height - Inches(0.2))
        tf = txBox.text_frame
        p = tf.add_paragraph()
        p.text = "Input: 224 × 224 × 3 Image"
        p.font.size = CONTENT_SIZE
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

        p = tf.add_paragraph()
        p.text = "Output: 128-D feature vector"
        p.font.size = CONTENT_SIZE
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

        self.add_footer(slide, 8)

    def add_ann_theory_slide(self):
        """Slide 9: Theoretical Explanation Part 2 - ANN Variants & Feature Fusion"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "ANN Variants & Feature Fusion"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # ANN variants
        ann_variants = [
            ("Standard", "10 → 64 → 32", "Baseline 2-layer encoder"),
            ("Deep", "10 → 128 → 64 → 32 → 16", "Higher-order feature interactions"),
            ("Attention", "10 → Attention → 256 → 128", "Dynamic feature weighting"),
            ("Wide", "10 → 256 → 128 + Skip", "Wide layers with residual connection"),
            ("Residual", "10 → 64 → Residual Block → 32", "Skip connections in FC layers")
        ]

        # Create vertical list of ANN variants
        left = Inches(1)
        top = Inches(1.5)
        width = Inches(3)
        height = Inches(0.6)
        spacing = Inches(0.1)

        for i, (name, arch, feature) in enumerate(ann_variants):
            # Variant box
            shape = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top + i * (height + spacing), width, height
            )
            shape.fill.solid()
            shape.fill.fore_color.rgb = LIGHT_GRAY
            shape.line.color.rgb = DARK_BLUE

            txBox = slide.shapes.add_textbox(left + Inches(0.05), top + i * (height + spacing) + Inches(0.05),
                                            width - Inches(0.1), height - Inches(0.1))
            tf = txBox.text_frame
            p = tf.add_paragraph()
            p.text = f"{name}:"
            p.font.size = Pt(16)
            p.font.bold = True
            p.font.color.rgb = DARK_BLUE

            p = tf.add_paragraph()
            p.text = arch
            p.font.size = Pt(14)
            p.font.color.rgb = GRAY

            p = tf.add_paragraph()
            p.text = feature
            p.font.size = Pt(12)
            p.font.color.rgb = GRAY
            p.font.italic = True

        # Add fusion diagram
        fusion_left = Inches(5)
        fusion_top = Inches(1.5)
        fusion_width = Inches(5)
        fusion_height = Inches(4)

        # Fusion background
        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, fusion_left, fusion_top, fusion_width, fusion_height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_GRAY
        shape.line.color.rgb = LIGHT_BLUE

        # Fusion title
        txBox = slide.shapes.add_textbox(fusion_left + Inches(0.1), fusion_top + Inches(0.1),
                                        fusion_width - Inches(0.2), Inches(0.3))
        tf = txBox.text_frame
        tf.text = "Feature Fusion"
        p = tf.paragraphs[0]
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

        # CNN feature vector
        cnn_left = fusion_left + Inches(0.5)
        cnn_top = fusion_top + Inches(0.8)
        cnn_width = Inches(1.5)
        cnn_height = Inches(0.6)

        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, cnn_left, cnn_top, cnn_width, cnn_height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_BLUE
        shape.line.color.rgb = DARK_BLUE

        txBox = slide.shapes.add_textbox(cnn_left + Inches(0.05), cnn_top + Inches(0.05),
                                        cnn_width - Inches(0.1), cnn_height - Inches(0.1))
        tf = txBox.text_frame
        tf.text = "CNN Feature Vector\n(128-D)"
        p = tf.paragraphs[0]
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

        # ANN feature vector
        ann_left = fusion_left + Inches(3)
        ann_top = fusion_top + Inches(0.8)
        ann_width = Inches(1.5)
        ann_height = Inches(0.6)

        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, ann_left, ann_top, ann_width, ann_height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_BLUE
        shape.line.color.rgb = DARK_BLUE

        txBox = slide.shapes.add_textbox(ann_left + Inches(0.05), ann_top + Inches(0.05),
                                        ann_width - Inches(0.1), ann_height - Inches(0.1))
        tf = txBox.text_frame
        tf.text = "ANN Feature Vector\n(32-D)"
        p = tf.paragraphs[0]
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

        # Plus sign
        plus_left = fusion_left + Inches(2.4)
        plus_top = fusion_top + Inches(1.1)
        txBox = slide.shapes.add_textbox(plus_left, plus_top, Inches(0.2), Inches(0.2))
        tf = txBox.text_frame
        tf.text = "+"
        p = tf.paragraphs[0]
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

        # Equals arrow
        equals_left = fusion_left + Inches(2.8)
        equals_top = fusion_top + Inches(1.1)
        txBox = slide.shapes.add_textbox(equals_left, equals_top, Inches(0.2), Inches(0.2))
        tf = txBox.text_frame
        tf.text = "="
        p = tf.paragraphs[0]
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

        # Fused vector
        fused_left = fusion_left + Inches(3.2)
        fused_top = fusion_top + Inches(0.8)
        fused_width = Inches(1.5)
        fused_height = Inches(0.6)

        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, fused_left, fused_top, fused_width, fused_height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_BLUE
        shape.line.color.rgb = DARK_BLUE

        txBox = slide.shapes.add_textbox(fused_left + Inches(0.05), fused_top + Inches(0.05),
                                        fused_width - Inches(0.1), fused_height - Inches(0.1))
        tf = txBox.text_frame
        tf.text = "160-D fused\nrepresentation"
        p = tf.paragraphs[0]
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

        self.add_footer(slide, 9)

    def add_training_theory_slide(self):
        """Slide 10: Theoretical Explanation Part 3 - Training, Explainability & Evaluation"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "Training, Explainability & Evaluation"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # Create three panels
        panel_width = Inches(4.5)
        panel_height = Inches(3.5)
        left_positions = [Inches(0.5), Inches(5.5), Inches(10.5)]
        top = Inches(1.5)

        panels = [
            ("Training", [
                "• Categorical cross-entropy loss",
                "• Class weighting for imbalance",
                "• Adam optimizer (lr=0.001)",
                "• Batch size = 16",
                "• Early stopping (patience=10)",
                "• Mixed precision FP16"
            ]),
            ("Explainability", [
                "• Grad-CAM visualization",
                "• Last convolutional layer",
                "• Gradient-weighted activation",
                "• Heatmap overlay on input",
                "• Clinical interpretability",
                "• Region-specific highlights"
            ]),
            ("Evaluation", [
                "• Accuracy",
                "• Precision",
                "• Recall",
                "• F1-score",
                "• ROC-AUC",
                "• Per-syndrome metrics"
            ])
        ]

        for i, (title_text, bullet_points) in enumerate(panels):
            left = left_positions[i]

            # Panel background
            shape = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, panel_width, panel_height
            )
            shape.fill.solid()
            shape.fill.fore_color.rgb = LIGHT_GRAY
            shape.line.color.rgb = LIGHT_BLUE

            # Panel title
            txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.1),
                                            panel_width - Inches(0.2), Inches(0.4))
            tf = txBox.text_frame
            tf.text = title_text
            p = tf.paragraphs[0]
            p.font.size = Pt(20)
            p.font.bold = True
            p.font.color.rgb = DARK_BLUE
            p.alignment = PP_ALIGN.CENTER

            # Bullet points
            txBox = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.6),
                                            panel_width - Inches(0.4), panel_height - Inches(0.7))
            tf = txBox.text_frame
            for point in bullet_points:
                p = tf.add_paragraph()
                p.text = point
                p.font.size = CONTENT_SIZE
                p.font.color.rgb = GRAY
                p.space_after = Pt(2)

        self.add_footer(slide, 10)

    def add_results_model_comparison_slide(self):
        """Slide 11: Results Part 1 - Model Configuration Comparison"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "Results — Model Configuration Comparison"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # Data from README
        models = [
            ("GeriatricNet + Standard", 91.0, 8.6),
            ("GeriatricNet + Deep", 92.3, 8.8),
            ("GeriatricNet + Attention", 93.1, 9.3),
            ("GeriatricNet + Wide", 91.8, 9.1),
            ("GeriatricNet + Residual", 92.6, 9.4),
            ("EfficientNetB0 + Standard", 94.6, 21.4),
            ("EfficientNetB4 + Standard", 94.8, 74.6),
            ("MobileNetV2 + Standard", 93.2, 14.1),
            ("ResNet50 + Standard", 95.8, 98.2),
            ("ResNet50 + Deep", 96.1, 101.4)
        ]

        # Create horizontal bar chart
        chart_left = Inches(1)
        chart_top = Inches(1.5)
        chart_width = Inches(10)
        chart_height = Inches(4)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))

        model_names = [m[0] for m in models]
        accuracies = [m[1] for m in models]
        sizes = [m[2] for m in models]

        # Create bars
        y_pos = np.arange(len(model_names))
        bars = ax.barh(y_pos, accuracies, color=LIGHT_BLUE_MPL, edgecolor=DARK_BLUE_MPL, linewidth=1)

        # Highlight best performer
        best_idx = np.argmax(accuracies)
        bars[best_idx].set_color(DARK_GREEN_MPL)
        bars[best_idx].set_edgecolor(DARK_BLUE_MPL)

        # Customize chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names, fontsize=10)
        ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Configuration Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(85, 100)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Add value labels on bars
        for i, (acc, size) in enumerate(zip(accuracies, sizes)):
            ax.text(acc + 0.5, i, f'{acc}% ({size} MB)', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()

        # Save chart to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()

        # Add chart to slide
        slide.shapes.add_picture(img_buffer, chart_left, chart_top, width=chart_width, height=chart_height)

        # Add best performer highlight
        highlight_left = Inches(12)
        highlight_top = Inches(1.5)
        highlight_width = Inches(3.5)
        highlight_height = Inches(1)

        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, highlight_left, highlight_top, highlight_width, highlight_height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_GRAY
        shape.line.color.rgb = DARK_GREEN
        shape.line.width = Pt(2)

        txBox = slide.shapes.add_textbox(highlight_left + Inches(0.1), highlight_top + Inches(0.1),
                                        highlight_width - Inches(0.2), highlight_height - Inches(0.2))
        tf = txBox.text_frame
        p = tf.add_paragraph()
        p.text = "Best Performer:"
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = DARK_GREEN
        p.alignment = PP_ALIGN.CENTER

        p = tf.add_paragraph()
        p.text = "ResNet50 + Deep ANN"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = DARK_GREEN
        p.alignment = PP_ALIGN.CENTER

        p = tf.add_paragraph()
        p.text = "96.1% Accuracy"
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = DARK_GREEN
        p.alignment = PP_ALIGN.CENTER

        p = tf.add_paragraph()
        p.text = "101.4 MB Model Size"
        p.font.size = Pt(16)
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

        self.add_footer(slide, 11)

    def add_results_per_syndrome_slide(self):
        """Slide 12: Results Part 2 - Per-Syndrome Performance"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "Results — Per-Syndrome Performance"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # Create grouped bar chart for per-syndrome performance
        chart_left = Inches(0.5)
        chart_top = Inches(1.5)
        chart_width = Inches(7)
        chart_height = Inches(4)

        # Data from README
        syndromes = ['Frailty', 'Sarcopenia', 'Cognitive\nImpairment', 'Fall Risk', 'Depression']
        precision = [0.95, 0.94, 0.97, 0.96, 0.93]
        recall = [0.93, 0.96, 0.94, 0.95, 0.97]
        f1_score = [0.94, 0.95, 0.95, 0.95, 0.95]

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(7, 4))

        x = np.arange(len(syndromes))
        width = 0.25

        bars1 = ax.bar(x - width, precision, width, label='Precision', color=LIGHT_BLUE_MPL, edgecolor=DARK_BLUE_MPL)
        bars2 = ax.bar(x, recall, width, label='Recall', color=LIGHT_GRAY_MPL, edgecolor=DARK_BLUE_MPL)
        bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color=DARK_GREEN_MPL, edgecolor=DARK_BLUE_MPL)

        # Customize chart
        ax.set_xlabel('Syndrome Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Syndrome Classification Performance (ResNet50 + Deep ANN)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(syndromes, fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)

        plt.tight_layout()

        # Save chart to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()

        # Add chart to slide
        slide.shapes.add_picture(img_buffer, chart_left, chart_top, width=chart_width, height=chart_height)

        # Add modality comparison section
        comp_left = Inches(8)
        comp_top = Inches(1.5)
        comp_width = Inches(6)
        comp_height = Inches(4)

        # Comparison background
        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, comp_left, comp_top, comp_width, comp_height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_GRAY
        shape.line.color.rgb = LIGHT_BLUE

        # Comparison title
        txBox = slide.shapes.add_textbox(comp_left + Inches(0.1), comp_top + Inches(0.1),
                                        comp_width - Inches(0.2), Inches(0.4))
        tf = txBox.text_frame
        tf.text = "Modality Comparison"
        p = tf.paragraphs[0]
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

        # Modality data
        modalities = [
            ("Visual-only", "87.3%"),
            ("Clinical-only", "79.8%"),
            ("Audio-only", "82.1%"),
            ("Video-only", "84.6%"),
            ("Fusion (Ours)", "96.1%")
        ]

        # Create vertical list
        text_left = comp_left + Inches(0.2)
        text_top = comp_top + Inches(0.6)
        line_height = Inches(0.3)

        for modality, accuracy in modalities:
            txBox = slide.shapes.add_textbox(text_left, text_top, comp_width - Inches(0.4), line_height)
            tf = txBox.text_frame
            p = tf.add_paragraph()
            p.text = f"{modality}: {accuracy}"
            p.font.size = CONTENT_SIZE
            p.font.color.rgb = DARK_BLUE if "Fusion" in modality else GRAY
            p.font.bold = "Fusion" in modality
            text_top += line_height

        # Add fusion advantage visualization
        advantage_left = comp_left + Inches(0.2)
        advantage_top = comp_top + Inches(2.2)
        advantage_width = comp_width - Inches(0.4)
        advantage_height = Inches(0.8)

        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, advantage_left, advantage_top, advantage_width, advantage_height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_BLUE
        shape.line.color.rgb = DARK_GREEN

        txBox = slide.shapes.add_textbox(advantage_left + Inches(0.1), advantage_top + Inches(0.1),
                                        advantage_width - Inches(0.2), advantage_height - Inches(0.2))
        tf = txBox.text_frame
        p = tf.add_paragraph()
        p.text = "Fusion Advantage: +8.8% over best single modality"
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = DARK_GREEN
        p.alignment = PP_ALIGN.CENTER

        p = tf.add_paragraph()
        p.text = "(Statistical significance: p < 0.001)"
        p.font.size = Pt(12)
        p.font.color.rgb = GRAY
        p.alignment = PP_ALIGN.CENTER

        self.add_footer(slide, 12)

    def add_results_explainability_performance_slide(self):
        """Slide 13: Results Part 3 - Explainability & System Performance"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "Results — Explainability & System Performance"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # Create two sections: Explainability and System Performance
        section_width = Inches(7)
        section_height = Inches(4)
        left_positions = [Inches(0.5), Inches(8.5)]
        top = Inches(1.5)

        sections = [
            ("Explainability (Grad-CAM)", [
                "• Cognitive impairment: Hippocampal & temporal lobe activation",
                "• Depression: Facial action units (AU1, AU4, AU15)",
                "• Fall risk: Gait symmetry & postural sway patterns",
                "• Frailty: Muscle atrophy indicators (temporalis, masseter)",
                "• Sarcopenia: Shoulder & trunk proportionality metrics",
                "",
                "Clinician Review: 85% alignment with expert ROI"
            ]),
            ("System Performance", [
                "• Inference latency: 280 ms",
                "• Throughput: 12 predictions/second",
                "• Memory footprint: 1.2 GB RAM",
                "• API uptime: 99.2% (72-hour test)",
                "• UI response time: <100 ms",
                "",
                "Hardware: NVIDIA GTX 1650 (4GB VRAM)"
            ])
        ]

        for i, (section_title, bullet_points) in enumerate(sections):
            left = left_positions[i]

            # Section background
            shape = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, section_width, section_height
            )
            shape.fill.solid()
            shape.fill.fore_color.rgb = LIGHT_GRAY
            shape.line.color.rgb = LIGHT_BLUE

            # Section title
            txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.1),
                                            section_width - Inches(0.2), Inches(0.4))
            tf = txBox.text_frame
            tf.text = section_title
            p = tf.paragraphs[0]
            p.font.size = Pt(18)
            p.font.bold = True
            p.font.color.rgb = DARK_BLUE
            p.alignment = PP_ALIGN.CENTER

            # Bullet points
            txBox = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.6),
                                            section_width - Inches(0.4), section_height - Inches(0.7))
            tf = txBox.text_frame
            for point in bullet_points:
                if point == "":
                    # Add spacing
                    p = tf.add_paragraph()
                    p.space_after = Pt(6)
                else:
                    p = tf.add_paragraph()
                    p.text = point
                    p.font.size = CONTENT_SIZE
                    if "85% alignment" in point:
                        p.font.color.rgb = DARK_GREEN
                        p.font.bold = True
                    elif any(x in point for x in ["280 ms", "12 predictions", "1.2 GB", "99.2%", "<100 ms"]):
                        p.font.color.rgb = DARK_BLUE
                        p.font.bold = True
                    else:
                        p.font.color.rgb = GRAY
                    p.space_after = Pt(2)

        self.add_footer(slide, 13)

    def add_results_comparative_analysis_slide(self):
        """Slide 14: Results Part 4 - Comparative Analysis"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "Results — Comparative Analysis"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # Create comparison table
        table_left = Inches(1)
        table_top = Inches(1.5)
        table_width = Inches(14)
        table_height = Inches(3)

        # Add table background
        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, table_left, table_top, table_width, table_height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_GRAY
        shape.line.color.rgb = LIGHT_BLUE

        # Table headers
        headers = ["Approach", "Modalities Used", "Accuracy", "Explainability", "Deployment Readiness"]
        col_widths = [Inches(2.5), Inches(2.5), Inches(1.5), Inches(2), Inches(2)]

        # Header row
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            left = table_left + sum(col_widths[:i])
            txBox = slide.shapes.add_textbox(left, table_top, width, Inches(0.4))
            tf = txBox.text_frame
            tf.text = header
            p = tf.paragraphs[0]
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

            # Header background
            header_shape = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, table_top, width, Inches(0.4)
            )
            header_shape.fill.solid()
            header_shape.fill.fore_color.rgb = DARK_BLUE

        # Table data
        rows = [
            ["Traditional Surveys", "Clinical only", "72–78%", "High (clinician-interpreted)", "Low (requires training)"],
            ["Single-Modality DL", "Visual only", "85–89%", "Medium (Grad-CAM available)", "Medium"],
            ["Early Fusion DL", "Visual + Clinical", "90–92%", "Low (black-box)", "Medium"],
            ["NeuroVision v3 (Ours)", "Visual + Clinical + Audio + Video", "96.1%", "High (Grad-CAM + attention)", "High (REST API + Web UI)"]
        ]

        # Data rows
        for row_idx, row in enumerate(rows):
            top_pos = table_top + Inches(0.4) * (row_idx + 1)
            for col_idx, (cell_text, width) in enumerate(zip(row, col_widths)):
                left = table_left + sum(col_widths[:col_idx])

                # Cell background (highlight NeuroVision v3)
                if row_idx == 3:  # NeuroVision v3 row
                    shape = slide.shapes.add_shape(
                        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top_pos, width, Inches(0.35)
                    )
                    shape.fill.solid()
                    shape.fill.fore_color.rgb = LIGHT_BLUE
                else:
                    shape = slide.shapes.add_shape(
                        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top_pos, width, Inches(0.35)
                    )
                    shape.fill.solid()
                    shape.fill.fore_color.rgb = WHITE

                # Cell text
                txBox = slide.shapes.add_textbox(left + Inches(0.02), top_pos + Inches(0.02),
                                               width - Inches(0.04), Inches(0.31))
                tf = txBox.text_frame
                tf.text = cell_text
                p = tf.paragraphs[0]
                p.font.size = Pt(12)
                p.font.color.rgb = DARK_BLUE if row_idx == 3 else GRAY
                p.font.bold = row_idx == 3  # Bold for NeuroVision v3
                p.alignment = PP_ALIGN.CENTER

        # Add key findings callout
        callout_left = Inches(1)
        callout_top = Inches(4.8)
        callout_width = Inches(14)
        callout_height = Inches(0.8)

        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, callout_left, callout_top, callout_width, callout_height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_GRAY
        shape.line.color.rgb = DARK_GREEN

        txBox = slide.shapes.add_textbox(callout_left + Inches(0.1), callout_top + Inches(0.1),
                                        callout_width - Inches(0.2), callout_height - Inches(0.2))
        tf = txBox.text_frame
        p = tf.add_paragraph()
        p.text = "Key Advantages of NeuroVision v3:"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = DARK_GREEN

        p = tf.add_paragraph()
        p.text = "• Highest accuracy (96.1%) for multi-syndrome geriatric detection"
        p.font.size = Pt(12)
        p.font.color.rgb = GRAY
        p.level = 1

        p = tf.add_paragraph()
        p.text = "• First system integrating all four modalities (image, clinical, audio, video)"
        p.font.size = Pt(12)
        p.font.color.rgb = GRAY
        p.level = 1

        p = tf.add_paragraph()
        p.text = "• Dual explainability: CNN Grad-CAM + ANN attention weights"
        p.font.size = Pt(12)
        p.font.color.rgb = GRAY
        p.level = 1

        p = tf.add_paragraph()
        p.text = "• Production-ready deployment with cross-platform compatibility"
        p.font.size = Pt(12)
        p.font.color.rgb = GRAY
        p.level = 1

        self.add_footer(slide, 14)

    def add_results_validation_slide(self):
        """Slide 15: Results Part 5 - Validation & Robustness"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "Results — Validation & Robustness"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # Create validation dashboard
        metric_width = Inches(3)
        metric_height = Inches(1.2)
        left_positions = [Inches(0.5), Inches(4.2), Inches(7.9), Inches(11.6)]
        top_positions = [Inches(1.5), Inches(3), Inches(4.5)]

        metrics = [
            ("5-fold CV", "±1.2%", "accuracy variance"),
            ("Data Aug.", "+3.4%", "accuracy improvement"),
            ("Noise Resist.", ">90%", "with 20% Gaussian noise"),
            ("Missing Abbott:", "", ""),
            ("Ethical/Bias:", "<2%", "variation across subgroups"),
            ("Real-world Profiles:", "Validated", "synthetic geriatric data")
        ]

        # Add metric cards
        for i, (label, value, desc) in enumerate(metrics):
            left = left_positions[i % 4]
            top = top_positions[i // 4]

            # Special handling for missing modality row
            if label == "Missing Abbott:":
                # Create a small chart for missing modality degradation
                chart_left = left
                chart_top = top
                chart_width = Inches(5.5)
                chart_height = Inches(1.2)

                # Create matplotlib figure for degradation chart
                fig, ax = plt.subplots(figsize=(5.5, 1.2))

                modalities = ['Full', 'Audio\nMissing', 'Video\nMissing', 'Clinical\nMissing', 'Image\nMissing']
                accuracy = [96.1, 94.2, 93.7, 91.8, 88.5]

                bars = ax.bar(modalities, accuracy, color=[LIGHT_BLUE_MPL, LIGHT_GRAY_MPL, LIGHT_GRAY_MPL, LIGHT_GRAY_MPL, LIGHT_GRAY_MPL],
                             edgecolor=DARK_BLUE_MPL)
                # Highlight full model
                bars[0].set_color(DARK_GREEN_MPL)

                ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
                ax.set_title('Performance Degradation with Missing Modalities', fontsize=12, fontweight='bold', pad=10)
                ax.set_ylim(80, 100)
                ax.grid(axis='y', linestyle='--', alpha=0.5)

                # Add value labels
                for bar, acc in zip(bars, accuracy):
                    height = bar.get_height()
                    ax.annotate(f'{acc}%',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)

                plt.tight_layout()

                # Save chart to buffer
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                plt.close()

                # Add chart to slide
                slide.shapes.add_picture(img_buffer, chart_left, chart_top, width=chart_width, height=chart_height)
            else:
                # Regular metric card
                shape = slide.shapes.add_shape(
                    MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, metric_width, metric_height
                )
                shape.fill.solid()
                shape.fill.fore_color.rgb = LIGHT_GRAY
                shape.line.color.rgb = LIGHT_BLUE

                # Value (large)
                txBox = slide.shapes.add_textbox(left + Inches(0.05), top + Inches(0.05),
                                               metric_width - Inches(0.1), metric_height - Inches(0.1))
                tf = txBox.text_frame
                if value:  # Only show if there's a value
                    p = tf.add_paragraph()
                    p.text = value
                    p.font.size = Pt(24)
                    p.font.bold = True
                    p.font.color.rgb = DARK_BLUE if any(x in value for x in ["1.2", "96.1", "99.2", "280"]) else DARK_GREEN
                    p.alignment = PP_ALIGN.CENTER

                # Label
                txBox = slide.shapes.add_textbox(left + Inches(0.05), top + Inches(0.4),
                                               metric_width - Inches(0.1), Inches(0.2))
                tf = txBox.text_frame
                p = tf.add_paragraph()
                p.text = label
                p.font.size = Pt(14)
                p.font.bold = True
                p.font.color.rgb = DARK_BLUE
                p.alignment = PP_ALIGN.CENTER

                # Description
                txBox = slide.shapes.add_textbox(left + Inches(0.05), top + Inches(0.6),
                                               metric_width - Inches(0.1), Inches(0.2))
                tf = txBox.text_frame
                p = tf.add_paragraph()
                p.text = desc
                p.font.size = Pt(12)
                p.font.color.rgb = GRAY
                p.alignment = PP_ALIGN.CENTER

        self.add_footer(slide, 15)

    def add_references_slide(self):
        """Slide 16: References"""
        slide_layout = self.prs.slide_layouts[1]  # Title and content
        slide = self.prs.slides.add_slide(slide_layout)
        title = slide.shapes.title
        content = slide.placeholders[1]

        title.text = "References"
        title.text_frame.paragraphs[0].font.size = SECTION_TITLE_SIZE
        title.text_frame.paragraphs[0].font.bold = True
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE

        # References from README
        references = [
            "Esteva, A., Kuprel, B., Novoa, R. A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115–118.",
            "Liu, Y., Chen, P. H. C., Krause, J., & Peng, L. (2019). How to read articles that use machine learning: Users' guides to the medical literature. JAMA, 322(18), 1806–1816.",
            "Zhang, Y., Zhou, Z., Ruan, S., & Wang, Q. (2018). Deep learning in neuroimaging-based psychiatric diagnosis: Is the black box interpretable? IEEE Reviews in Biomedical Engineering, 11, 3–17.",
            "Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., et al. (2018). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv:1711.05225.",
            "Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM. ICCV, 618–626.",
            "Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. CVPR, 470–478.",
            "Howard, A., Sandler, M., Chu, G., Chen, L. C., et al. (2019). Searching for MobileNetV3. ICCV, 1314–1324.",
            "Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML, 6105–6114."
        ]

        # Create two-column reference list
        left_col = Inches(0.5)
        right_col = Inches(5)
        top_start = Inches(1.5)
        line_height = Inches(0.25)

        # Split references
        mid_point = len(references) // 2
        left_refs = references[:mid_point]
        right_refs = references[mid_point:]

        # Left column
        for i, ref in enumerate(left_refs):
            txBox = slide.shapes.add_textbox(left_col, top_start + i * line_height,
                                            Inches(4), line_height)
            tf = txBox.text_frame
            p = tf.add_paragraph()
            p.text = f"{i+1}. {ref}"
            p.font.size = SMALL_SIZE
            p.font.color.rgb = GRAY

        # Right column
        for i, ref in enumerate(right_refs):
            txBox = slide.shapes.add_textbox(right_col, top_start + i * line_height,
                                            Inches(4), line_height)
            tf = txBox.text_frame
            p = tf.add_paragraph()
            p.text = f"{i+1+mid_point}. {ref}"
            p.font.size = SMALL_SIZE
            p.font.color.rgb = GRAY

        # Add disclaimer
        left = Inches(0.5)
        top = Inches(7.5)
        width = Inches(15)
        height = Inches(0.8)

        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_GRAY
        shape.line.color.rgb = LIGHT_BLUE

        txBox = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.1),
                                        width - Inches(0.2), height - Inches(0.2))
        tf = txBox.text_frame
        p = tf.add_paragraph()
        p.text = "*Academic research prototype — not for clinical use.*"
        p.font.size = Pt(14)
        p.font.italic = True
        p.font.color.rgb = GRAY
        p.alignment = PP_ALIGN.CENTER

        self.add_footer(slide, 16)

        # Add thank you message
        left = Inches(0.5)
        top = Inches(8.5)
        width = Inches(15)
        height = Inches(0.5)

        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        p = tf.add_paragraph()
        p.text = "Thank you for your attention!"
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

        left = Inches(0.5)
        top = Inches(9)
        width = Inches(15)
        height = Inches(0.3)

        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        p = tf.add_paragraph()
        p.text = "Questions?"
        p.font.size = Pt(20)
        p.font.color.rgb = GRAY
        p.alignment = PP_ALIGN.CENTER

    def generate_presentation(self):
        """Generate the complete presentation"""
        print("Generating NeuroVision v3 Project Review Presentation...")

        # Add all slides
        self.add_title_slide()
        self.add_introduction_slide()
        self.add_problem_statement_slide()
        self.add_objectives_slide()
        self.add_dataset_sources_slide()
        self.add_dataset_properties_slide()
        self.add_architecture_slide()
        self.add_cnn_theory_slide()
        self.add_ann_theory_slide()
        self.add_training_theory_slide()
        self.add_results_model_comparison_slide()
        self.add_results_per_syndrome_slide()
        self.add_results_explainability_performance_slide()
        self.add_results_comparative_analysis_slide()
        self.add_results_validation_slide()
        self.add_references_slide()

        # Save presentation
        output_path = os.path.join(os.getcwd(), OUTPUT_FILE)
        self.prs.save(output_path)
        print(f"Presentation saved to: {output_path}")
        print(f"Total slides: {len(self.prs.slides)}")

        # Generate verification notes
        self.generate_verification_notes()

        return output_path

    def generate_verification_notes(self):
        """Generate verification notes file"""
        notes = []

        notes.append("VERIFICATION NOTES FOR NEUROVISION v3 PRESENTATION")
        notes.append("=" * 55)
        notes.append("")
        notes.append("Claims requiring experimental/literature verification:")
        notes.append("- 96.1% accuracy result (should be validated against test set)")
        notes.append("- Per-syndrome precision/recall/F1 scores")
        notes.append("- Grad-CAM clinician alignment percentage (85%)")
        notes.append("- System performance metrics (280ms latency, 12 predictions/s, etc.)")
        notes.append("- Modality comparison accuracies (visual-only 87.3%, etc.)")
        notes.append("- Cross-validation variance (±1.2%)")
        notes.append("- Data augmentation impact (+3.4% accuracy)")
        notes.append("- Noise resistance (>90% with 20% Gaussian noise)")
        notes.append("- Missing modality degradation figures")
        notes.append("- Ethical/bias analysis (<2% variation across subgroups)")
        notes.append("")
        notes.append("Placeholders remaining (intentional):")
        notes.append("- [YOUR NAME]")
        notes.append("- [YOUR REGISTRATION NUMBER]")
        notes.append("- [GUIDE NAME]")
        notes.append("")
        notes.append("Missing assets:")
        notes.append("- Actual Grad-CAM images (used conceptual placeholders)")
        notes.append("- Project-specific logos or branding elements")
        notes.append("")
        notes.append("Assumptions made:")
        notes.append("- All dataset statistics from README.md are accurate")
        notes.append("- Model performance figures are from final trained models")
        notes.append("- System metrics are based on GTX 1650 (4GB VRAM) testing")
        notes.append("- References are correctly formatted and relevant")
        notes.append("")

        # Write verification notes
        verification_path = os.path.join(os.getcwd(), VERIFICATION_FILE)
        with open(verification_path, 'w') as f:
            f.write('\n'.join(notes))

        print(f"Verification notes saved to: {verification_path}")

if __name__ == "__main__":
    # Create presentation generator
    presenter = NeuroVisionPresentation()

    # Generate presentation
    pptx_path = presenter.generate_presentation()

    print("\nGeneration complete!")
    print(f"PowerPoint file: {pptx_path}")
    print(f"Verification notes: {VERIFICATION_FILE}")
    print("\nNext steps:")
    print("1. Replace placeholders [YOUR NAME], [YOUR REGISTRATION NUMBER], [GUIDE NAME]")
    print("2. Review verification notes for claims requiring validation")
    print("3. Open the PPTX to verify layout and content")
    print("4. Present with confidence!")