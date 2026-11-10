"""Add two evidence-based result slides after the saved-run summary slide."""
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

SRC = "NeuroVision_v3_Project_Review_Updated_Fixed.pptx"
OUT = "NeuroVision_v3_Project_Review_Results_Expanded.pptx"
with open("trained_models/GeriatricNet_Standard_metrics.json", encoding="utf-8") as f:
    data = json.load(f)

prs = Presentation(SRC)
NAVY=RGBColor(11,46,79); BLUE=RGBColor(28,114,147); TEAL=RGBColor(50,145,161)
GOLD=RGBColor(235,164,57); INK=RGBColor(40,57,72); MUTED=RGBColor(104,124,140)
PAPER=RGBColor(249,250,248); WHITE=RGBColor(255,255,255); LINE=RGBColor(216,226,230)
PALE=RGBColor(237,244,245); GREEN=RGBColor(73,142,112)

def txt(s,x,y,w,h,text,size=12,color=INK,bold=False,align=PP_ALIGN.LEFT):
    sh=s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf=sh.text_frame; tf.clear(); tf.word_wrap=True; tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    tf.margin_left=tf.margin_right=Inches(.04); tf.margin_top=tf.margin_bottom=Inches(.02)
    p=tf.paragraphs[0]; p.text=text; p.alignment=align
    for r in p.runs:
        r.font.name='Aptos'; r.font.size=Pt(size); r.font.bold=bold; r.font.color.rgb=color
    return sh

def roundbox(s,x,y,w,h,fill=WHITE,line=LINE,rect=False):
    sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE if rect else MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=fill; sh.line.color.rgb=line; sh.line.width=Pt(1)
    return sh

def header(s,kicker,title,subtitle,page):
    s.background.fill.solid(); s.background.fill.fore_color.rgb=PAPER
    txt(s,.68,.28,10,.3,kicker.upper(),10,BLUE,True)
    txt(s,.68,.67,18.5,.65,title,29,NAVY,True)
    txt(s,.7,1.38,18,.35,subtitle,13,MUTED)
    txt(s,18.3,10.82,1,.2,f'{page} / 20',9,MUTED,False,PP_ALIGN.RIGHT)

# Slide 15: class-level metrics from the saved JSON.
cl=prs.slides.add_slide(prs.slide_layouts[6]); header(cl,'Results','Performance by syndrome','Precision, recall, and F1-score from the saved validation run.',15)
txt(cl,.95,1.95,4,.35,'SYNDROME',10,BLUE,True)
for x,label,col in [(6.05,'PRECISION',BLUE),(10.3,'RECALL',TEAL),(14.55,'F1-SCORE',GOLD)]: txt(cl,x,1.95,3.8,.35,label,10,NAVY,True,PP_ALIGN.CENTER)
for i,item in enumerate(data['per_class']):
    y=2.45+i*1.22; roundbox(cl,.82,y,18.1,.98,WHITE if i%2==0 else PALE,LINE)
    txt(cl,1.05,y+.16,4.25,.56,item['syndrome'],14,NAVY,True)
    for x,key,col in [(6.05,'precision',BLUE),(10.3,'recall',TEAL),(14.55,'f1_score',GOLD)]:
        v=float(item[key]); roundbox(cl,x,y+.28,2.55,.16,RGBColor(225,232,235),RGBColor(225,232,235),True)
        roundbox(cl,x,y+.28,max(.04,2.55*v),.16,col,col,True)
        txt(cl,x+2.7,y+.1,.9,.5,f'{100*v:.1f}%',12,INK,True,PP_ALIGN.RIGHT)
roundbox(cl,.85,8.95,18,.72,RGBColor(255,244,224),RGBColor(239,218,180))
txt(cl,1.08,9.05,17.5,.48,'Recall differs across classes; these values come from one saved validation run and need further validation.',12,NAVY,True,PP_ALIGN.CENTER)

# Slide 16: recorded training and validation histories, plotted directly from JSON.
cv=prs.slides.add_slide(prs.slide_layouts[6]); header(cv,'Results','Training curves','Accuracy and loss across the 15 epochs recorded for this model.',16)
h=data['history']
def plot(s,x,y,w,hgt,title,a,b,vmin,vmax,ca,cb):
    txt(s,x,y-.45,w,.32,title,15,NAVY,True)
    left=x+.65; top=y+.08; pw=w-.9; ph=hgt-.72
    roundbox(s,left,top,pw,ph,WHITE,LINE,True)
    for k in range(5):
        yy=top+ph-k*ph/4
        gl=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(left),Inches(yy),Inches(left+pw),Inches(yy))
        gl.line.color.rgb=RGBColor(232,237,239); gl.line.width=Pt(.7)
        txt(s,x,yy-.09,.52,.18,f'{vmin+(vmax-vmin)*k/4:.1f}',8,MUTED,False,PP_ALIGN.RIGHT)
    n=len(a)
    for i in [0,4,9,14]: txt(s,left+pw*i/(n-1)-.2,top+ph+.02,.4,.18,str(i+1),8,MUTED,False,PP_ALIGN.CENTER)
    for vals,col in [(a,ca),(b,cb)]:
        pts=[(left+pw*i/(n-1),top+ph-(v-vmin)/(vmax-vmin)*ph) for i,v in enumerate(vals)]
        for (x1,y1),(x2,y2) in zip(pts,pts[1:]):
            seg=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
            seg.line.color.rgb=col; seg.line.width=Pt(2.2)
        for xx,yy in pts:
            dot=s.shapes.add_shape(MSO_SHAPE.OVAL,Inches(xx-.035),Inches(yy-.035),Inches(.07),Inches(.07))
            dot.fill.solid(); dot.fill.fore_color.rgb=col; dot.line.color.rgb=col
    for j,(lab,col) in enumerate([('Training',ca),('Validation',cb)]):
        lx=x+.9+j*1.7; ly=y+hgt-.08
        dot=s.shapes.add_shape(MSO_SHAPE.OVAL,Inches(lx),Inches(ly),Inches(.12),Inches(.12)); dot.fill.solid(); dot.fill.fore_color.rgb=col; dot.line.color.rgb=col
        txt(s,lx+.18,ly-.04,1.3,.2,lab,9,MUTED)
plot(cv,.85,2.5,8.9,5.25,'Accuracy',h['train_acc'],h['val_acc'],0,.8,BLUE,GOLD)
plot(cv,10.1,2.5,8.9,5.25,'Loss',h['train_loss'],h['val_loss'],1,2,GREEN,GOLD)
roundbox(cv,.85,8.35,18,.98,RGBColor(255,244,224),RGBColor(239,218,180))
txt(cv,1.08,8.45,17.5,.72,'A confusion matrix cannot be made from the saved files because the per-example validation labels and predictions were not stored. Do not show the app’s confusion-matrix view as a measured result from this run.',11,NAVY,True,PP_ALIGN.CENTER)

# Move appended slides to after slide 14 in presentation order.
ids=prs.slides._sldIdLst
children=list(ids)
new_ids=children[-2:]
for el in new_ids: ids.remove(el)
for offset,el in enumerate(new_ids): ids.insert(14+offset,el)

# Update page counters on the preserved tail slides after the insertion.
for slide_num,slide in enumerate(list(prs.slides)[16:],17):
    old=f'{slide_num-2} / 18'
    new=f'{slide_num} / 20'
    for sh in slide.shapes:
        if sh.has_text_frame and old in sh.text:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    if old in r.text: r.text=r.text.replace(old,new)

# The saved-run summary is also part of the expanded 20-slide deck.
for sh in prs.slides[13].shapes:
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if "/ 18" in r.text:
                    r.text = r.text.replace("/ 18", "/ 20")

prs.save(OUT)
print(OUT)
