from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

OUT = Path("tmp/pdfs/NeuroVision_Project_Review_Notes_Source.pptx")
OUT.parent.mkdir(parents=True, exist_ok=True)

prs = Presentation()
prs.slide_width = Inches(8.5)
prs.slide_height = Inches(11)
blank = prs.slide_layouts[6]

NAVY=RGBColor(14,47,76)
TEAL=RGBColor(37,126,148)
GOLD=RGBColor(225,157,49)
INK=RGBColor(43,57,69)
MUTED=RGBColor(99,117,132)
PAPER=RGBColor(250,250,247)
PALE=RGBColor(235,243,245)
PALE2=RGBColor(246,247,243)
WHITE=RGBColor(255,255,255)
LINE=RGBColor(216,226,230)
GREEN=RGBColor(62,132,103)


def box(slide,x,y,w,h,fill=WHITE,line=LINE):
    sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=fill
    sh.line.color.rgb=line; sh.line.width=Pt(.8)
    return sh


def text(slide,x,y,w,h,value,size=10,color=INK,bold=False,font="Aptos",align=PP_ALIGN.LEFT):
    sh=slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf=sh.text_frame; tf.clear(); tf.word_wrap=True
    tf.margin_left=Inches(.02); tf.margin_right=Inches(.02)
    tf.margin_top=Inches(.015); tf.margin_bottom=Inches(.01)
    tf.vertical_anchor=MSO_ANCHOR.TOP
    for idx,line in enumerate(value.split("\n")):
        p=tf.paragraphs[0] if idx==0 else tf.add_paragraph()
        p.text=line; p.alignment=align; p.space_after=Pt(2)
        for r in p.runs:
            r.font.name=font; r.font.size=Pt(size); r.font.color.rgb=color; r.font.bold=bold
    return sh


def header(slide,tag,title,subtitle,page):
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb=PAPER
    bar=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(8.5), Inches(.13))
    bar.fill.solid(); bar.fill.fore_color.rgb=TEAL; bar.line.fill.background()
    text(slide,.55,.35,7.35,.22,tag.upper(),9,TEAL,True)
    text(slide,.55,.66,7.4,.58,title,22,NAVY,True,"Georgia")
    text(slide,.57,1.27,7.3,.40,subtitle,10,MUTED)
    text(slide,.55,10.64,6.8,.20,"NEUROVISION  |  PROJECT REVIEW NOTES",8,MUTED,True)
    text(slide,7.25,10.64,.65,.20,str(page),8,MUTED,False,align=PP_ALIGN.RIGHT)


def section(slide,x,y,w,h,title,body,accent=TEAL,fill=WHITE,body_size=10):
    box(slide,x,y,w,h,fill,LINE)
    stripe=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y+.14), Inches(.055), Inches(h-.28))
    stripe.fill.solid(); stripe.fill.fore_color.rgb=accent; stripe.line.fill.background()
    text(slide,x+.18,y+.14,w-.34,.28,title,12,NAVY,True)
    text(slide,x+.18,y+.49,w-.35,h-.58,body,body_size,INK)


# Page 1 - simple project explanation
s=prs.slides.add_slide(blank)
header(s,"Quick brief","NeuroVision in simple words","A short explanation you can use when introducing the project.",1)
section(s,.55,1.88,7.4,1.15,"What the project is",
        "NeuroVision is a research prototype that explores image and clinical-feature inputs to estimate five health-risk categories in older adults.",TEAL,WHITE,11)
section(s,.55,3.20,7.4,2.15,"The saved training model",
        "Image branch: 224 x 224 x 3 image -> GeriatricNet CNN -> 128 image features\nClinical branch: 10 generated values -> Standard ANN (64 -> 32) -> 32 clinical features\nFusion: 128 + 32 = 160 features -> Dense 256 -> Dense 128 -> Softmax with 5 outputs",GOLD,PALE,10)
section(s,.55,5.54,7.4,1.68,"The five output classes",
        "1. Frailty\n2. Sarcopenia\n3. Cognitive Impairment\n4. Fall Risk\n5. Depression",GREEN,WHITE,10)
section(s,.55,7.39,7.4,1.12,"A safe one-sentence summary",
        "It is an experimental software prototype, not a medical diagnostic tool. The saved training run reached 68.67% validation accuracy for one model configuration.",TEAL,PALE2,10)
text(s,.62,8.78,7.18,1.15,
     "Say this if asked about modalities: The application has audio and video routes, but the saved training run uses prepared images plus generated clinical features. It does not train a four-modality fusion model.",
     10,NAVY,True)


# Page 2 - precise code line map
s=prs.slides.add_slide(blank)
header(s,"Code map","Where the main parts live","Use these file and line references when your reviewer asks how something works.",2)
items=[
    ("Inputs, feature count, and class names","backend/app/models/hybrid_model.py:44-64\nImage is 224 x 224 x 3; 10 clinical inputs; 5 class labels."),
    ("Standard ANN","backend/app/models/hybrid_model.py:71-96\n10 inputs -> Dense 64 -> Dense 32; outputs a 32-value feature vector."),
    ("GeriatricNet CNN","backend/app/models/hybrid_model.py:339-429\nCustom image branch; its final feature vector is 128-D."),
    ("Fusion and output layer","backend/app/models/hybrid_model.py:545-631\nInputs are concatenated; final Dense layer uses softmax with 5 classes."),
    ("Which model was trained","backend/train_all.py:76-78\nOnly custom + standard is in TRAINING_CONFIGS: GeriatricNet_Standard."),
    ("Dataset loading and generated clinical data","backend/train_all.py:89-159\nReads prepared image folders; creates clinical values in code; does not load CSVs here."),
    ("Train/validation split","backend/train_all.py:383-400\nShuffled 80/20 split with random seed 42; one configuration is run."),
    ("Training and saved metrics","backend/train_all.py:269-355\nCallbacks, validation evaluation, per-class scores, curves, JSON metrics."),
    ("Grad-CAM","backend/app/models/grad_cam.py:68-149\nHeatmap calculation; overlay helper at lines 152-175."),
    ("Prediction API","backend/app/routes/predict.py:254-324\nPrediction, class scores, confidence indicator, and optional Grad-CAM."),
    ("App entry points","backend/main.py:18-44; frontend/src/services/apiService.ts:224-310\nFastAPI router and frontend API/mock fallback logic."),
]
y=1.85
for idx,(title,body) in enumerate(items):
    h=.72 if idx not in (5,7) else .84
    section(s,.55,y,7.4,h,title,body,[TEAL,GOLD,GREEN][idx%3],WHITE,8.6)
    y += h+.07


# Page 3 - datasets and training
s=prs.slides.add_slide(blank)
header(s,"Data and method","What the training run actually used","Separate prepared training images from other files stored in dataset/.",3)
section(s,.55,1.82,7.4,1.05,"Prepared image folders on disk (current counts)",
        "Frailty 800 | Sarcopenia 800 | Cognitive Impairment 800 | Fall Risk 396 | Depression 800\nTotal in prepared folders: 3,596 images. The training loader caps samples per class (train_all.py:59-66, 135-140), so folder totals are not automatically the exact run sample count.",TEAL,WHITE,9)
section(s,.55,3.02,7.4,1.45,"How image labels are assigned",
        "process_raw_data.py:63-103 maps fall images to Fall Risk and FER2013 expressions to other classes: neutral/disgust -> Frailty; happy/surprise -> Sarcopenia; angry -> Cognitive Impairment; sad/fear -> Depression. These are proxy labels, not clinical diagnoses.",GOLD,PALE,9)
section(s,.55,4.62,7.4,1.13,"Clinical inputs in the training code",
        "train_all.py:89-114 and 148-157 generates 10 numeric clinical values from class-specific distributions, then normalizes them. They are not read from the CSV files for this saved run.",GREEN,WHITE,9)
section(s,.55,5.88,7.4,1.13,"Other files in dataset/",
        "The folder also contains CSV and audio files. The saved training loader does not consume those CSV/audio files. Audio/video model code and API routes exist, but they are not included in TRAINING_CONFIGS (train_all.py:76-78).",TEAL,WHITE,9)
section(s,.55,7.14,7.4,1.18,"Image processing and split",
        "Images are resized and scaled to 0-1; training augmentation uses horizontal flips, brightness, and contrast (train_all.py:240-267). The code combines source FER train/test folders during processing, then makes a random row-level 80/20 split (process_raw_data.py:74-103; train_all.py:383-386). Do not describe it as an independent patient-level test set.",GOLD,PALE2,9)


# Page 4 - saved results with a clear source
s=prs.slides.add_slide(blank)
header(s,"Results","The saved result to quote","These values come from trained_models/GeriatricNet_Standard_metrics.json.",4)
box(s,.55,1.85,7.4,1.38,NAVY,NAVY)
text(s,.8,2.02,6.9,.28,"GERIATRICNET + STANDARD ANN",11,RGBColor(205,226,235),True,align=PP_ALIGN.CENTER)
text(s,.8,2.37,6.9,.64,"68.67% validation accuracy",21,GOLD,True,"Georgia",PP_ALIGN.CENTER)
rows=[
    ("AUC (logged)","0.916"),("Saved model size","9.7 MB"),("Recorded epochs","15"),
    ("Overall precision (logged)","94.95%"),("Overall recall (logged)","31.33%"),
]
y=3.48
for label,val in rows:
    box(s,.55,y,7.4,.48,WHITE,LINE)
    text(s,.72,y+.08,5.35,.28,label,9,INK,True)
    text(s,6.12,y+.08,1.5,.28,val,10,NAVY,True,align=PP_ALIGN.RIGHT)
    y+=.56
section(s,.55,6.38,7.4,1.02,"Per-class F1-score",
        "Frailty 53.0% | Sarcopenia 52.3% | Cognitive Impairment 64.7% | Fall Risk 86.3% | Depression 87.3%",TEAL,PALE,9)
section(s,.55,7.55,7.4,1.15,"Important difference in the app",
        "The frontend contains a hard-coded 91.5% demo metric (apiService.ts:97-111). It is not the saved training result. If asked, explain the discrepancy and quote 68.67% from the saved metrics file. Avoid presenting the app's mock confusion matrix as real evaluation evidence.",GOLD,PALE2,9)


# Page 5 - common viva questions and short answers
s=prs.slides.add_slide(blank)
header(s,"Review Q&A","Short answers to likely questions","Keep answers direct. Point to the listed file and line when asked for proof.",5)
qa=[
    ("Which file defines the five output classes?","hybrid_model.py:46, 52-58. The model output layer is at 623."),
    ("What are the model inputs?","A 224 x 224 x 3 image and 10 clinical values. See hybrid_model.py:44-46, 579-580."),
    ("Where are the clinical values from?","They are generated in train_all.py:89-114, not loaded from the CSV files in the saved run."),
    ("Were ten models trained?","No. The current training config lists one: GeriatricNet + Standard ANN (train_all.py:76-78). Other architectures are code options."),
    ("Why does the app show 91.5%?","That is a hard-coded demo metric in frontend/src/services/apiService.ts:97-111. The saved metrics JSON reports 68.67%."),
    ("Is the confidence a calibrated uncertainty score?","No. The API calculates a simple score from model output values (predict.py:299-306); it has not been calibrated as clinical uncertainty."),
    ("Is Grad-CAM implemented?","The Grad-CAM code exists (grad_cam.py:68-149) and the prediction endpoint attempts it with an image and a loaded model (predict.py:275-288). Show a real heatmap only if one is produced."),
    ("Can I show a confusion matrix?","Not as a measured saved result: the metrics JSON has no per-example labels/predictions or confusion matrix. The UI has demo matrices."),
    ("Can it diagnose a patient?","No. It is an academic research prototype and should not be used for diagnosis or treatment decisions."),
]
y=1.85
for idx,(q,a) in enumerate(qa):
    h=.82
    box(s,.55,y,7.4,h,WHITE,LINE)
    text(s,.72,y+.08,7.02,.25,q,10,NAVY,True)
    text(s,.72,y+.37,7.0,.37,a,8.6,INK)
    y+=.91

prs.save(OUT)
print(OUT)
