def create_geriatricnet_svg():
    width = 1920
    height = 800

    svg = []
    svg.append(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <filter id="cardShadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.06"/>
      <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.04"/>
    </filter>

    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#0284c7" />
    </marker>

    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&amp;display=swap');
      text {{ font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
      .title {{ font-size: 26px; font-weight: 800; fill: #0f172a; }}
      .subtitle {{ font-size: 15px; font-weight: 500; fill: #64748b; }}
      .block-title {{ font-size: 16px; font-weight: 800; fill: #0f172a; }}
      .block-desc {{ font-size: 12.5px; font-weight: 500; fill: #334155; line-height: 1.4; }}
      .dim-badge {{ font-size: 12px; font-weight: 700; fill: #ffffff; }}
      .tag-lbl {{ font-size: 11px; font-weight: 700; fill: #0284c7; letter-spacing: 0.5px; text-transform: uppercase; }}
    </style>
  </defs>

  <rect width="{width}" height="{height}" fill="#f8fafc" />
  <rect x="0" y="0" width="{width}" height="6" fill="#0284c7" />

  <!-- Header -->
  <g transform="translate(60, 50)">
    <text x="0" y="0" class="title">Fig. 2. GeriatricNet Convolutional Backbone Architecture</text>
    <text x="0" y="28" class="subtitle">Custom Lightweight Deep CNN with Hierarchical Feature Maps, Depthwise Separable Bottleneck, and Global Average Pooling Head</text>
  </g>
''')

    # 6 stages from left to right:
    # 1. Input Image
    # 2. Block 1
    # 3. Block 2
    # 4. Block 3
    # 5. Block 4 & Depthwise
    # 6. Feature Projection Head
    
    stages = [
        {
            "x": 60, "w": 250, "h": 500,
            "tag": "Input Stage",
            "title": "Input Image",
            "desc": [
                "• 224 × 224 × 3 RGB Tensor",
                "• Normalized to [0, 1]",
                "• Patient Posture &amp; Facial Cues",
                "• Batch Size: 32"
            ],
            "pill": "224 × 224 × 3",
            "color": "#0284c7",
            "bg": "#ffffff",
            "stroke": "#bae6fd"
        },
        {
            "x": 360, "w": 260, "h": 500,
            "tag": "Hierarchical Block 1",
            "title": "Low-Level Edges",
            "desc": [
                "• Conv2D (32 filters, 3×3)",
                "• Batch Normalization",
                "• ReLU Activation",
                "• Conv2D (32 filters, 3×3)",
                "• Batch Normalization + ReLU",
                "• MaxPool (2×2, stride 2)",
                "───",
                "Boundary &amp; Contour Focus"
            ],
            "pill": "112 × 112 × 32",
            "color": "#0284c7",
            "bg": "#ffffff",
            "stroke": "#bae6fd"
        },
        {
            "x": 670, "w": 260, "h": 500,
            "tag": "Hierarchical Block 2",
            "title": "Texture & Patterns",
            "desc": [
                "• Conv2D (64 filters, 3×3)",
                "• Batch Normalization",
                "• ReLU Activation",
                "• Conv2D (64 filters, 3×3)",
                "• Batch Normalization + ReLU",
                "• MaxPool (2×2, stride 2)",
                "• Spatial Dropout (0.20)",
                "───",
                "Facial &amp; Muscle Textures"
            ],
            "pill": "56 × 56 × 64",
            "color": "#0284c7",
            "bg": "#ffffff",
            "stroke": "#bae6fd"
        },
        {
            "x": 980, "w": 260, "h": 500,
            "tag": "Hierarchical Block 3",
            "title": "Semantic Motifs",
            "desc": [
                "• Conv2D (128 filters, 3×3)",
                "• Batch Normalization",
                "• ReLU Activation",
                "• Conv2D (128 filters, 3×3)",
                "• Batch Normalization + ReLU",
                "• MaxPool (2×2, stride 2)",
                "───",
                "Postural Slouch &amp; Asymmetry"
            ],
            "pill": "28 × 28 × 128",
            "color": "#0284c7",
            "bg": "#ffffff",
            "stroke": "#bae6fd"
        },
        {
            "x": 1290, "w": 270, "h": 500,
            "tag": "Lightweight Bottleneck",
            "title": "Depthwise Separable",
            "desc": [
                "• Conv2D (256 filters, 3×3) + BN",
                "• DepthwiseConv2D (3×3)",
                "  (Spatial per-channel filter)",
                "• PointwiseConv2D (1×1, 256)",
                "  (Cross-channel projection)",
                "• Factorized Ratio R ≈ 1/K²",
                "• MaxPool (2×2, stride 2)",
                "───",
                "8% Size of ResNet50"
            ],
            "pill": "14 × 14 × 256",
            "color": "#0369a1",
            "bg": "#f0f9ff",
            "stroke": "#38bdf8"
        },
        {
            "x": 1610, "w": 250, "h": 500,
            "tag": "Projection Head",
            "title": "Visual Feature Head",
            "desc": [
                "• Global Average Pooling 2D",
                "  (Converts spatial map to 256-d)",
                "• Dense Layer (256 neurons)",
                "• Batch Normalization + ReLU",
                "• Dropout (rate = 0.40)",
                "• Dense Layer (128 neurons)",
                "───",
                "Ready for Fusion Stage"
            ],
            "pill": "Output: 128-d Vector",
            "color": "#1d4ed8",
            "bg": "#eff6ff",
            "stroke": "#2563eb"
        }
    ]

    y_top = 130
    for i, s in enumerate(stages):
        x, w, h = s["x"], s["w"], s["h"]
        svg.append(f'''
    <!-- Stage {i+1} -->
    <g filter="url(#cardShadow)">
      <rect x="{x}" y="{y_top}" width="{w}" height="{h}" rx="14" fill="{s["bg"]}" stroke="{s["stroke"]}" stroke-width="2"/>
      <rect x="{x}" y="{y_top}" width="8" height="{h}" rx="4" fill="{s["color"]}"/>
    </g>
    <text x="{x+25}" y="{y_top+32}" class="tag-lbl" fill="{s["color"]}">{s["tag"]}</text>
    <text x="{x+25}" y="{y_top+58}" class="block-title">{s["title"]}</text>
    <line x1="{x+25}" y1="{y_top+72}" x2="{x+w-25}" y2="{y_top+72}" stroke="#f1f5f9" stroke-width="2"/>
        ''')

        # Desc lines
        dy = y_top + 102
        for line in s["desc"]:
            svg.append(f'<text x="{x+25}" y="{dy}" class="block-desc">{line}</text>')
            dy += 26

        # Dimension pill badge at bottom
        svg.append(f'''
    <rect x="{x+25}" y="{y_top+h-52}" width="{w-50}" height="32" rx="8" fill="{s["color"]}"/>
    <text x="{x+w/2}" y="{y_top+h-31}" text-anchor="middle" class="dim-badge">{s["pill"]}</text>
        ''')

        # Arrow to next stage
        if i < len(stages) - 1:
            next_x = stages[i+1]["x"]
            svg.append(f'''
    <line x1="{x+w}" y1="{y_top+h/2}" x2="{next_x-8}" y2="{y_top+h/2}" stroke="#0284c7" stroke-width="3" marker-end="url(#arrow)"/>
            ''')

    # Footer Notes
    svg.append(f'''
  <!-- Bottom info bar -->
  <rect x="60" y="660" width="{width-120}" height="80" rx="10" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="url(#cardShadow)"/>
  <text x="85" y="694" font-size="13.5" font-weight="700" fill="#0f172a">Computational Characteristics of GeriatricNet:</text>
  <text x="85" y="720" font-size="13" font-weight="500" fill="#475569">• Total Parameters: 2.1M (~8% of ResNet50's 24.9M)  |  • Saved Size: 8.6 MB  |  • CPU Inference Latency: 8.4 ms  |  • Validation Accuracy: 91.5%</text>
  <text x="{width-85}" y="708" text-anchor="end" font-size="13" font-weight="700" fill="#0284c7">Canva-Ready Vector Diagram • NeuroVision / GeroFusion Net</text>
</svg>
''')
    return "".join(svg)

svg_data = create_geriatricnet_svg()
with open("GeriatricNet_Architecture_Canva.svg", "w", encoding="utf-8") as f:
    f.write(svg_data)
print("GeriatricNet_Architecture_Canva.svg created!")
