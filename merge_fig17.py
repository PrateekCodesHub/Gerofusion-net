import os
from PIL import Image, ImageDraw, ImageFont

p1 = r'C:/Users/manis/.gemini/antigravity/brain/c8b79036-5d21-4dbb-a11a-70ffad0dbf5e/.user_uploaded/media_1794302806530_a050a2ec.jpg'
p2 = r'C:/Users/manis/.gemini/antigravity/brain/c8b79036-5d21-4dbb-a11a-70ffad0dbf5e/.user_uploaded/media_1794302816035_7e2db494.jpg'

im1 = Image.open(p1).convert('RGB')
im2 = Image.open(p2).convert('RGB')

# Try to load Arial or default font
font_large = None
font_sub = None
try:
    font_large = ImageFont.truetype("arialbd.ttf", 26)
    font_sub = ImageFont.truetype("arial.ttf", 20)
except:
    font_large = ImageFont.load_default()
    font_sub = ImageFont.load_default()

# -------------------------------------------------------------------------
# 1. STACKED COMPOSITE (Top / Bottom) - Highly Recommended for Journals
# -------------------------------------------------------------------------
padding = 30
header_h = 45
w = 1024
total_w = w + (padding * 2)
total_h = padding + header_h + im1.height + padding + header_h + im2.height + padding

stacked = Image.new('RGB', (total_w, total_h), color=(248, 250, 252)) # #f8fafc slate bg
draw_s = ImageDraw.Draw(stacked)

# Draw Panel (a)
y_a = padding
draw_s.text((padding + 5, y_a + 8), "(a) Multi-Modal Media Upload & Image Input Stage", font=font_large, fill=(15, 23, 42))
stacked.paste(im1, (padding, y_a + header_h))
# Add subtle border around image
draw_s.rectangle([padding, y_a + header_h, padding + w, y_a + header_h + im1.height], outline=(203, 213, 225), width=2)

# Draw Panel (b)
y_b = y_a + header_h + im1.height + padding
draw_s.text((padding + 5, y_b + 8), "(b) Hybrid CNN-ANN Phenotyping Report & Risk Prediction", font=font_large, fill=(15, 23, 42))
stacked.paste(im2, (padding, y_b + header_h))
draw_s.rectangle([padding, y_b + header_h, padding + w, y_b + header_h + im2.height], outline=(203, 213, 225), width=2)

out_stacked_path = "output/report_figures/Figure_17_Image_Input_and_Prediction_Output_Stacked.png"
stacked.save(out_stacked_path, dpi=(300, 300))
stacked.save("Figure_17_Image_Input_and_Prediction_Output.png", dpi=(300, 300))
print(f"Saved stacked composite: {out_stacked_path}")

# -------------------------------------------------------------------------
# 2. SIDE-BY-SIDE COMPOSITE (Horizontal)
# -------------------------------------------------------------------------
# Standardize heights to 691 px (matching im1)
target_h = 691
# Resize im2 keeping aspect ratio or letterbox
im2_scaled = im2.resize((int(im2.width * (target_h / im2.height)), target_h), Image.Resampling.LANCZOS)

side_w = padding + im1.width + padding + im2_scaled.width + padding
side_h = padding + header_h + target_h + padding

side_by_side = Image.new('RGB', (side_w, side_h), color=(248, 250, 252))
draw_side = ImageDraw.Draw(side_by_side)

# Left panel (a)
draw_side.text((padding + 5, padding + 8), "(a) Image Input Stage", font=font_large, fill=(15, 23, 42))
side_by_side.paste(im1, (padding, padding + header_h))
draw_side.rectangle([padding, padding + header_h, padding + im1.width, padding + header_h + im1.height], outline=(203, 213, 225), width=2)

# Right panel (b)
x_b = padding + im1.width + padding
draw_side.text((x_b + 5, padding + 8), "(b) Prediction Report", font=font_large, fill=(15, 23, 42))
side_by_side.paste(im2_scaled, (x_b, padding + header_h))
draw_side.rectangle([x_b, padding + header_h, x_b + im2_scaled.width, padding + header_h + im2_scaled.height], outline=(203, 213, 225), width=2)

out_side_path = "output/report_figures/Figure_17_Image_Input_and_Prediction_Output_SideBySide.png"
side_by_side.save(out_side_path, dpi=(300, 300))
print(f"Saved side-by-side composite: {out_side_path}")
