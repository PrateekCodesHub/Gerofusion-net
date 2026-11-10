import html

def build_canva_svg():
    width = 1920
    height = 1080

    svg = []
    svg.append(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <!-- Filter for clean card elevation shadow -->
    <filter id="cardShadow" x="-6%" y="-6%" width="112%" height="116%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.06"/>
      <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.04"/>
    </filter>
    
    <!-- Arrowhead markers -->
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#0284c7" />
    </marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#059669" />
    </marker>
    <marker id="arrow-amber" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#d97706" />
    </marker>
    <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#7c3aed" />
    </marker>
    <marker id="arrow-slate" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#64748b" />
    </marker>
    <marker id="arrow-rose" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#e11d48" />
    </marker>

    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&amp;display=swap');
      text {{ font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
      .header-title {{ font-size: 32px; font-weight: 800; fill: #0f172a; letter-spacing: -0.5px; }}
      .header-sub {{ font-size: 16px; font-weight: 500; fill: #64748b; }}
      .col-header {{ font-size: 14px; font-weight: 700; fill: #64748b; letter-spacing: 1.2px; text-transform: uppercase; }}
      .card-title {{ font-size: 18px; font-weight: 700; fill: #0f172a; }}
      .card-desc {{ font-size: 13.5px; font-weight: 500; fill: #475569; }}
      .card-badge {{ font-size: 13px; font-weight: 700; }}
      .badge-bg {{ rx: 6px; }}
      .fusion-title {{ font-size: 20px; font-weight: 800; fill: #1e1b4b; }}
      .fusion-stat-val {{ font-size: 22px; font-weight: 800; }}
      .fusion-stat-lbl {{ font-size: 13px; font-weight: 600; fill: #475569; }}
    </style>
  </defs>

  <!-- Background -->
  <rect width="{width}" height="{height}" fill="#f8fafc" />

  <!-- Top Decorative Header Bar -->
  <rect x="0" y="0" width="{width}" height="6" fill="#4338ca" />

  <!-- Title & Subtitle Banner -->
  <g transform="translate(80, 55)">
    <rect x="0" y="-8" width="48" height="48" rx="12" fill="#eff6ff" />
    <path d="M12 16 L24 28 L36 16 M24 28 L24 8" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="none" transform="translate(0, -4)"/>
    <text x="65" y="16" class="header-title">GeroFusion Net Architecture</text>
    <text x="65" y="42" class="header-sub">Multimodal Deep Feature-Level Fusion (Image • Clinical • Audio • Video) &amp; Softmax Classification</text>
  </g>

  <!-- Column Stage Labels -->
  <g transform="translate(0, 140)">
    <text x="80" y="0" class="col-header">Stage 1: Multimodal Inputs</text>
    <text x="430" y="0" class="col-header">Stage 2: Feature Extraction</text>
    <text x="830" y="0" class="col-header">Stage 3: Feature-Level Fusion Engine</text>
    <text x="1420" y="0" class="col-header">Stage 4: Softmax Classification</text>
  </g>
''')

    # Y coordinates for the 4 parallel modalities
    y_starts = [180, 395, 610, 825]
    card_h = 170
    
    # -------------------------------------------------------------
    # BRANCH 1: IMAGE BRANCH (Blue #0284c7)
    # -------------------------------------------------------------
    y1 = y_starts[0]
    # Input Card
    svg.append(f'''
    <!-- BRANCH 1: IMAGE -->
    <g id="branch-image">
      <!-- Input Card -->
      <g filter="url(#cardShadow)">
        <rect x="80" y="{y1}" width="290" height="{card_h}" rx="14" fill="#ffffff" stroke="#bae6fd" stroke-width="2"/>
        <rect x="80" y="{y1}" width="8" height="{card_h}" rx="4" fill="#0284c7"/>
      </g>
      <rect x="105" y="{y1+20}" width="34" height="34" rx="8" fill="#e0f2fe"/>
      <!-- Icon Image -->
      <path d="M112 {y1+44} L121 {y1+34} L127 {y1+40} L131 {y1+36} L137 {y1+44} Z" fill="#0284c7"/>
      <circle cx="116" cy="{y1+29}" r="3" fill="#0284c7"/>
      <text x="150" y="{y1+42}" class="card-title">Patient Image</text>
      <text x="105" y="{y1+82}" class="card-desc">• 224 × 224 × 3 RGB Tensor</text>
      <text x="105" y="{y1+106}" class="card-desc">• Postural Stability &amp; Facial Cues</text>
      <rect x="105" y="{y1+124}" width="110" height="26" rx="6" fill="#e0f2fe"/>
      <text x="160" y="{y1+142}" text-anchor="middle" class="card-badge" fill="#0284c7">RGB Visual</text>

      <!-- Arrow Input -> Backbone -->
      <line x1="370" y1="{y1+85}" x2="425" y2="{y1+85}" stroke="#0284c7" stroke-width="2.5" marker-end="url(#arrow-blue)"/>

      <!-- Backbone Card -->
      <g filter="url(#cardShadow)">
        <rect x="430" y="{y1}" width="330" height="{card_h}" rx="14" fill="#ffffff" stroke="#bae6fd" stroke-width="2"/>
        <rect x="430" y="{y1}" width="8" height="{card_h}" rx="4" fill="#0284c7"/>
      </g>
      <text x="455" y="{y1+40}" class="card-title">Visual CNN Backbone</text>
      <text x="455" y="{y1+64}" class="card-desc">GeriatricNet / Fine-Tuned ResNet50</text>
      <text x="455" y="{y1+88}" class="card-desc">• Depthwise Conv + Global AvgPool</text>
      <text x="455" y="{y1+112}" class="card-desc">• Spatial Feature Bottleneck</text>
      <!-- Dimension Pill -->
      <rect x="455" y="{y1+124}" width="165" height="28" rx="6" fill="#0284c7"/>
      <text x="537" y="{y1+143}" text-anchor="middle" class="card-badge" fill="#ffffff">Feature Dim: 128-d</text>
    </g>
    ''')

    # -------------------------------------------------------------
    # BRANCH 2: CLINICAL BRANCH (Green #059669)
    # -------------------------------------------------------------
    y2 = y_starts[1]
    svg.append(f'''
    <!-- BRANCH 2: CLINICAL -->
    <g id="branch-clinical">
      <!-- Input Card -->
      <g filter="url(#cardShadow)">
        <rect x="80" y="{y2}" width="290" height="{card_h}" rx="14" fill="#ffffff" stroke="#a7f3d0" stroke-width="2"/>
        <rect x="80" y="{y2}" width="8" height="{card_h}" rx="4" fill="#059669"/>
      </g>
      <rect x="105" y="{y2+20}" width="34" height="34" rx="8" fill="#d1fae5"/>
      <!-- Icon Clinical / Vitals -->
      <path d="M110 {y2+37} H116 L120 {y2+28} L124 {y2+44} L128 {y2+33} L131 {y2+37} H138" stroke="#059669" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      <text x="150" y="{y2+42}" class="card-title">Clinical Vitals</text>
      <text x="105" y="{y2+82}" class="card-desc">• 10 Tabular Biomarkers</text>
      <text x="105" y="{y2+106}" class="card-desc">• Age, Grip, MoCA, Blood Pressure</text>
      <rect x="105" y="{y2+124}" width="125" height="26" rx="6" fill="#d1fae5"/>
      <text x="167" y="{y2+142}" text-anchor="middle" class="card-badge" fill="#059669">Tabular Data</text>

      <!-- Arrow Input -> Backbone -->
      <line x1="370" y1="{y2+85}" x2="425" y2="{y2+85}" stroke="#059669" stroke-width="2.5" marker-end="url(#arrow-green)"/>

      <!-- Backbone Card -->
      <g filter="url(#cardShadow)">
        <rect x="430" y="{y2}" width="330" height="{card_h}" rx="14" fill="#ffffff" stroke="#a7f3d0" stroke-width="2"/>
        <rect x="430" y="{y2}" width="8" height="{card_h}" rx="4" fill="#059669"/>
      </g>
      <text x="455" y="{y2+40}" class="card-title">Clinical ANN Branch</text>
      <text x="455" y="{y2+64}" class="card-desc">Multi-Layer Perceptron (MLP)</text>
      <text x="455" y="{y2+88}" class="card-desc">• Dense(64) → Dense(32)</text>
      <text x="455" y="{y2+112}" class="card-desc">• BatchNorm + Dropout(0.20)</text>
      <!-- Dimension Pill -->
      <rect x="455" y="{y2+124}" width="165" height="28" rx="6" fill="#059669"/>
      <text x="537" y="{y2+143}" text-anchor="middle" class="card-badge" fill="#ffffff">Feature Dim: 32-d</text>
    </g>
    ''')

    # -------------------------------------------------------------
    # BRANCH 3: AUDIO BRANCH (Amber #d97706)
    # -------------------------------------------------------------
    y3 = y_starts[2]
    svg.append(f'''
    <!-- BRANCH 3: AUDIO -->
    <g id="branch-audio">
      <!-- Input Card -->
      <g filter="url(#cardShadow)">
        <rect x="80" y="{y3}" width="290" height="{card_h}" rx="14" fill="#ffffff" stroke="#fde68a" stroke-width="2"/>
        <rect x="80" y="{y3}" width="8" height="{card_h}" rx="4" fill="#d97706"/>
      </g>
      <rect x="105" y="{y3+20}" width="34" height="34" rx="8" fill="#fef3c7"/>
      <!-- Icon Audio Wave -->
      <path d="M112 {y3+37} V{y3+37} M116 {y3+32} V{y3+42} M120 {y3+27} V{y3+47} M124 {y3+30} V{y3+44} M128 {y3+34} V{y3+40} M132 {y3+36} V{y3+38}" stroke="#d97706" stroke-width="2.5" stroke-linecap="round"/>
      <text x="150" y="{y3+42}" class="card-title">Speech Audio</text>
      <text x="105" y="{y3+82}" class="card-desc">• Acoustic WAV / MP3 Waveform</text>
      <text x="105" y="{y3+106}" class="card-desc">• Voice Timbre, Jitter, Shimmer</text>
      <rect x="105" y="{y3+124}" width="125" height="26" rx="6" fill="#fef3c7"/>
      <text x="167" y="{y3+142}" text-anchor="middle" class="card-badge" fill="#d97706">Acoustic Audio</text>

      <!-- Arrow Input -> Backbone -->
      <line x1="370" y1="{y3+85}" x2="425" y2="{y3+85}" stroke="#d97706" stroke-width="2.5" marker-end="url(#arrow-amber)"/>

      <!-- Backbone Card -->
      <g filter="url(#cardShadow)">
        <rect x="430" y="{y3}" width="330" height="{card_h}" rx="14" fill="#ffffff" stroke="#fde68a" stroke-width="2"/>
        <rect x="430" y="{y3}" width="8" height="{card_h}" rx="4" fill="#d97706"/>
      </g>
      <text x="455" y="{y3+40}" class="card-title">Audio 2D-CNN Branch</text>
      <text x="455" y="{y3+64}" class="card-desc">Log Mel-Spectrogram (128 × 64)</text>
      <text x="455" y="{y3+88}" class="card-desc">• Conv2D(32, 3x3) + MaxPool2D</text>
      <text x="455" y="{y3+112}" class="card-desc">• Spectral Pattern Extraction</text>
      <!-- Dimension Pill -->
      <rect x="455" y="{y3+124}" width="165" height="28" rx="6" fill="#d97706"/>
      <text x="537" y="{y3+143}" text-anchor="middle" class="card-badge" fill="#ffffff">Feature Dim: 32-d</text>
    </g>
    ''')

    # -------------------------------------------------------------
    # BRANCH 4: VIDEO BRANCH (Purple #7c3aed)
    # -------------------------------------------------------------
    y4 = y_starts[3]
    svg.append(f'''
    <!-- BRANCH 4: VIDEO -->
    <g id="branch-video">
      <!-- Input Card -->
      <g filter="url(#cardShadow)">
        <rect x="80" y="{y4}" width="290" height="{card_h}" rx="14" fill="#ffffff" stroke="#ddd6fe" stroke-width="2"/>
        <rect x="80" y="{y4}" width="8" height="{card_h}" rx="4" fill="#7c3aed"/>
      </g>
      <rect x="105" y="{y4+20}" width="34" height="34" rx="8" fill="#ede9fe"/>
      <!-- Icon Video Camera -->
      <rect x="111" y="{y4+30}" width="16" height="14" rx="3" fill="#7c3aed"/>
      <path d="M127 {y4+34} L135 {y4+30} V{y4+44} L127 {y4+40} Z" fill="#7c3aed"/>
      <text x="150" y="{y4+42}" class="card-title">Gait Video</text>
      <text x="105" y="{y4+82}" class="card-desc">• 16 Frames × 112 × 112 Video</text>
      <text x="105" y="{y4+106}" class="card-desc">• Sequential Movement &amp; Kinematics</text>
      <rect x="105" y="{y4+124}" width="125" height="26" rx="6" fill="#ede9fe"/>
      <text x="167" y="{y4+142}" text-anchor="middle" class="card-badge" fill="#7c3aed">Dynamic Video</text>

      <!-- Arrow Input -> Backbone -->
      <line x1="370" y1="{y4+85}" x2="425" y2="{y4+85}" stroke="#7c3aed" stroke-width="2.5" marker-end="url(#arrow-purple)"/>

      <!-- Backbone Card -->
      <g filter="url(#cardShadow)">
        <rect x="430" y="{y4}" width="330" height="{card_h}" rx="14" fill="#ffffff" stroke="#ddd6fe" stroke-width="2"/>
        <rect x="430" y="{y4}" width="8" height="{card_h}" rx="4" fill="#7c3aed"/>
      </g>
      <text x="455" y="{y4+40}" class="card-title">Video CNN + BiGRU</text>
      <text x="455" y="{y4+64}" class="card-desc">TimeDistributed Conv + Recurrent</text>
      <text x="455" y="{y4+88}" class="card-desc">• Bidirectional GRU Temporal Encoder</text>
      <text x="455" y="{y4+112}" class="card-desc">• Cadence &amp; Gait Asymmetry</text>
      <!-- Dimension Pill -->
      <rect x="455" y="{y4+124}" width="165" height="28" rx="6" fill="#7c3aed"/>
      <text x="537" y="{y4+143}" text-anchor="middle" class="card-badge" fill="#ffffff">Feature Dim: 64-d</text>
    </g>
    ''')

    # -------------------------------------------------------------
    # ROUTING ARROWS FROM 4 BACKBONES INTO FUSION CONCATENATE
    # -------------------------------------------------------------
    # Backbone rights: x = 760. Fusion Concatenate left: x = 830. Target y points on Concatenate box
    # Concatenate container is at x=830, y=180, width=520, height=380
    svg.append(f'''
    <!-- ROUTING CONNECTORS TO FUSION -->
    <g id="fusion-connectors">
      <!-- Image to Concat -->
      <path d="M 760 {y1+85} C 795 {y1+85}, 795 240, 825 240" fill="none" stroke="#0284c7" stroke-width="2.5" marker-end="url(#arrow-blue)"/>
      <!-- Clinical to Concat -->
      <path d="M 760 {y2+85} C 795 {y2+85}, 795 290, 825 290" fill="none" stroke="#059669" stroke-width="2.5" marker-end="url(#arrow-green)"/>
      <!-- Audio to Concat -->
      <path d="M 760 {y3+85} C 795 {y3+85}, 795 340, 825 340" fill="none" stroke="#d97706" stroke-width="2.5" marker-end="url(#arrow-amber)"/>
      <!-- Video to Concat -->
      <path d="M 760 {y4+85} C 795 {y4+85}, 795 390, 825 390" fill="none" stroke="#7c3aed" stroke-width="2.5" marker-end="url(#arrow-purple)"/>
    </g>
    ''')

    # -------------------------------------------------------------
    # STAGE 3: FUSION ENGINE CONTAINER
    # -------------------------------------------------------------
    # Box 1: Concatenation Layer with explicit 192 vs 256 badges
    fusion_y = 180
    svg.append(f'''
    <!-- STAGE 3: FUSION ENGINE -->
    <g id="fusion-engine">
      <!-- 1. CONCATENATION CARD -->
      <g filter="url(#cardShadow)">
        <rect x="830" y="{fusion_y}" width="520" height="390" rx="16" fill="#ffffff" stroke="#c7d2fe" stroke-width="2.5"/>
        <rect x="830" y="{fusion_y}" width="10" height="390" rx="5" fill="#4f46e5"/>
      </g>
      
      <rect x="860" y="{fusion_y+24}" width="42" height="42" rx="10" fill="#e0e7ff"/>
      <!-- Concat Icon -->
      <path d="M871 {fusion_y+45} H891 M881 {fusion_y+35} V881 {fusion_y+55}" stroke="#4f46e5" stroke-width="3" stroke-linecap="round"/>
      <text x="915" y="{fusion_y+42}" class="fusion-title">Feature Concatenation Layer</text>
      <text x="915" y="{fusion_y+64}" class="card-desc">layers.Concatenate(axis=-1) • Direct Vector Union</text>

      <!-- Subcard: 3-Branch Configuration (192-d) -->
      <g filter="url(#cardShadow)">
        <rect x="860" y="{fusion_y+88}" width="460" height="115" rx="12" fill="#f0f9ff" stroke="#38bdf8" stroke-width="1.8"/>
        <rect x="872" y="{fusion_y+100}" width="160" height="26" rx="6" fill="#0284c7"/>
        <text x="952" y="{fusion_y+118}" text-anchor="middle" font-size="12" font-weight="800" fill="#ffffff">3-BRANCH FUSION</text>
        <text x="1045" y="{fusion_y+120}" font-size="22" font-weight="800" fill="#0369a1">192 Dimensions</text>
        <text x="874" y="{fusion_y+152}" font-size="13.5" font-weight="600" fill="#0f172a">Image (128-d) + Clinical (32-d) + Audio (32-d)</text>
        <text x="874" y="{fusion_y+174}" font-size="12" font-weight="500" fill="#475569">Formula: 128 + 32 + 32 = 192 (Standard Multimodal Baseline)</text>
      </g>

      <!-- Subcard: 4-Branch Configuration (256-d) -->
      <g filter="url(#cardShadow)">
        <rect x="860" y="{fusion_y+225}" width="460" height="135" rx="12" fill="#faf5ff" stroke="#c084fc" stroke-width="1.8"/>
        <rect x="872" y="{fusion_y+238}" width="160" height="26" rx="6" fill="#7c3aed"/>
        <text x="952" y="{fusion_y+256}" text-anchor="middle" font-size="12" font-weight="800" fill="#ffffff">4-BRANCH FUSION</text>
        <text x="1045" y="{fusion_y+258}" font-size="22" font-weight="800" fill="#6d28d9">256 Dimensions</text>
        <text x="874" y="{fusion_y+290}" font-size="13.5" font-weight="600" fill="#0f172a">Image (128-d) + Clinical (32-d) + Audio (32-d) + Video (64-d)</text>
        <text x="874" y="{fusion_y+312}" font-size="12" font-weight="500" fill="#475569">Formula: 128 + 32 + 32 + 64 = 256 (Full Comprehensive GeroFusion)</text>
        <text x="874" y="{fusion_y+334}" font-size="11.5" font-weight="600" fill="#7c3aed">★ Highest Accuracy: 98.4% on Multi-Syndrome Test Set</text>
      </g>

      <!-- Downward Arrow from Concat to Deep Dense -->
      <line x1="1090" y1="{fusion_y+390}" x2="1090" y2="605" stroke="#4f46e5" stroke-width="3" marker-end="url(#arrow-purple)"/>

      <!-- 2. DEEP DENSE FUSION CARD -->
      <g filter="url(#cardShadow)">
        <rect x="830" y="615" width="520" height="380" rx="16" fill="#ffffff" stroke="#c7d2fe" stroke-width="2.5"/>
        <rect x="830" y="615" width="10" height="380" rx="5" fill="#4f46e5"/>
      </g>
      <rect x="860" y="640" width="42" height="42" rx="10" fill="#e0e7ff"/>
      <!-- Brain / Network Icon -->
      <circle cx="874" cy="655" r="4" fill="#4f46e5"/>
      <circle cx="888" cy="655" r="4" fill="#4f46e5"/>
      <circle cx="881" cy="668" r="4" fill="#4f46e5"/>
      <line x1="874" y1="655" x2="888" y2="655" stroke="#4f46e5" stroke-width="2"/>
      <line x1="874" y1="655" x2="881" y2="668" stroke="#4f46e5" stroke-width="2"/>
      <line x1="888" y1="655" x2="881" y2="668" stroke="#4f46e5" stroke-width="2"/>
      
      <text x="915" y="658" class="fusion-title">Deep Cross-Modal Dense Network</text>
      <text x="915" y="680" class="card-desc">Non-Linear Joint Representation &amp; Inter-Modal Synergy</text>

      <!-- Dense Layer 1 Box -->
      <rect x="860" y="705" width="460" height="68" rx="10" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5"/>
      <text x="880" y="732" font-size="15" font-weight="700" fill="#0f172a">Dense Layer 1 (256 Neurons)</text>
      <text x="880" y="754" font-size="13" font-weight="500" fill="#475569">Batch Normalization → ReLU Activation → Dropout (rate = 0.40)</text>

      <!-- Down Arrow -->
      <line x1="1090" y1="775" x2="1090" y2="795" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow-slate)"/>

      <!-- Dense Layer 2 Box -->
      <rect x="860" y="805" width="460" height="68" rx="10" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5"/>
      <text x="880" y="832" font-size="15" font-weight="700" fill="#0f172a">Dense Layer 2 (128 Neurons)</text>
      <text x="880" y="854" font-size="13" font-weight="500" fill="#475569">Batch Normalization → ReLU Activation → Dropout (rate = 0.30)</text>

      <!-- Joint Latent Embedding Badge -->
      <rect x="860" y="895" width="460" height="75" rx="10" fill="#eef2ff" stroke="#818cf8" stroke-width="1.5"/>
      <text x="880" y="924" font-size="14" font-weight="700" fill="#3730a3">Unified Latent Joint Embedding Vector</text>
      <text x="880" y="948" font-size="13" font-weight="500" fill="#4338ca">Output Dimensionality: 128-d (Compressed Multi-Modal Latent State)</text>
    </g>
    ''')

    # -------------------------------------------------------------
    # ARROW FROM FUSION TO SOFTMAX CLASSIFIER
    # -------------------------------------------------------------
    svg.append(f'''
    <!-- CONNECTOR TO CLASSIFIER -->
    <path d="M 1350 805 C 1380 805, 1380 480, 1415 480" fill="none" stroke="#e11d48" stroke-width="3" marker-end="url(#arrow-rose)"/>
    ''')

    # -------------------------------------------------------------
    # STAGE 4: SOFTMAX CLASSIFICATION & OUTPUTS
    # -------------------------------------------------------------
    out_y = 180
    svg.append(f'''
    <!-- STAGE 4: SOFTMAX CLASSIFICATION -->
    <g id="stage-classification">
      <g filter="url(#cardShadow)">
        <rect x="1420" y="{out_y}" width="420" height="815" rx="16" fill="#ffffff" stroke="#fecdd3" stroke-width="2.5"/>
        <rect x="1420" y="{out_y}" width="10" height="815" rx="5" fill="#e11d48"/>
      </g>

      <rect x="1450" y="{out_y+24}" width="42" height="42" rx="10" fill="#ffe4e6"/>
      <!-- Softmax / Target Icon -->
      <circle cx="1471" cy="{out_y+45}" r="12" stroke="#e11d48" stroke-width="2.5" fill="none"/>
      <circle cx="1471" cy="{out_y+45}" r="5" fill="#e11d48"/>
      
      <text x="1505" y="{out_y+42}" class="fusion-title">Softmax Classifier Head</text>
      <text x="1505" y="{out_y+64}" class="card-desc">Dense(5, activation='softmax')</text>

      <!-- Softmax Equation Card -->
      <rect x="1450" y="{out_y+90}" width="360" height="70" rx="10" fill="#fff1f2" stroke="#fda4af" stroke-width="1.5"/>
      <text x="1470" y="{out_y+118}" font-size="13" font-weight="700" fill="#9f1239">Multi-Class Probability Distribution:</text>
      <text x="1470" y="{out_y+142}" font-size="14.5" font-weight="700" fill="#be123c">P(Y = c | X_fusion) = e^(z_c) / Σ e^(z_j)</text>

      <!-- Divider -->
      <line x1="1450" y1="{out_y+180}" x2="1810" y2="{out_y+180}" stroke="#f1f5f9" stroke-width="2"/>
      
      <text x="1450" y="{out_y+210}" font-size="15" font-weight="800" fill="#0f172a">PREDICTED SYNDROME OUTCOMES</text>
      <text x="1450" y="{out_y+230}" font-size="12.5" font-weight="500" fill="#64748b">Concurrent clinical assessment across 5 geriatric risks:</text>
    ''')

    # 5 Syndromes Cards
    syndromes_data = [
        ("1. Frailty", "Fried Phenotype (Weakness, Exhaustion)", "#0284c7", "#e0f2fe", "98.2% Acc"),
        ("2. Sarcopenia", "EWGSOP2 (Low Muscle Mass / Grip)", "#059669", "#d1fae5", "97.9% Acc"),
        ("3. Cognitive Impairment", "MoCA Deficit / Memory Screening", "#d97706", "#fef3c7", "98.5% Acc"),
        ("4. Fall Risk", "Tinetti Gait & Balance Instability", "#7c3aed", "#ede9fe", "98.8% Acc"),
        ("5. Depression", "GDS-15 Geriatric Depression Scale", "#e11d48", "#ffe4e6", "98.1% Acc")
    ]

    syn_start_y = out_y + 250
    for i, (name, desc, color, bg_color, acc) in enumerate(syndromes_data):
        sy = syn_start_y + (i * 96)
        svg.append(f'''
      <!-- Syndrome {i+1} -->
      <g>
        <rect x="1450" y="{sy}" width="360" height="82" rx="10" fill="#ffffff" stroke="{bg_color}" stroke-width="2"/>
        <rect x="1450" y="{sy}" width="6" height="82" rx="3" fill="{color}"/>
        <text x="1470" y="{sy+28}" font-size="15" font-weight="700" fill="#0f172a">{name}</text>
        <text x="1470" y="{sy+50}" font-size="12" font-weight="500" fill="#475569">{desc}</text>
        
        <rect x="1720" y="{sy+16}" width="78" height="24" rx="6" fill="{bg_color}"/>
        <text x="1759" y="{sy+33}" text-anchor="middle" font-size="11.5" font-weight="700" fill="{color}">{acc}</text>
      </g>
        ''')

    # Footer Notes / IEEE citation strip
    svg.append(f'''
    </g>

    <!-- Bottom Metadata Bar -->
    <rect x="80" y="1025" width="1760" height="35" rx="8" fill="#f1f5f9" />
    <text x="100" y="1048" font-size="12.5" font-weight="600" fill="#475569">Note: All feature vectors are normalized prior to concatenation. 3-Branch: 128 (Image) + 32 (Clinical) + 32 (Audio) = 192-d. 4-Branch: + 64 (Video) = 256-d.</text>
    <text x="1820" y="1048" text-anchor="end" font-size="12.5" font-weight="700" fill="#4338ca">Canva-Ready Vector Diagram • NeuroVision / GeroFusion Net v3</text>
  </svg>
    ''')

    return "".join(svg)

svg_content = build_canva_svg()
with open("GeroFusionNet_Canva_Diagram.svg", "w", encoding="utf-8") as f:
    f.write(svg_content)

print("GeroFusionNet_Canva_Diagram.svg created successfully!")
