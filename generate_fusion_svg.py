def create_fusion_svg():
    width = 1600
    height = 700

    svg = []
    svg.append(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="5" stdDeviation="6" flood-color="#0f172a" flood-opacity="0.06"/>
    </filter>
    <marker id="arr" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 9 5 L 0 9 z" fill="#4338ca" />
    </marker>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&amp;display=swap');
      text {{ font-family: 'Plus Jakarta Sans', -apple-system, sans-serif; }}
      .title {{ font-size: 24px; font-weight: 800; fill: #0f172a; }}
      .sub {{ font-size: 14px; font-weight: 500; fill: #64748b; }}
      .box-t {{ font-size: 15px; font-weight: 700; fill: #0f172a; }}
      .box-d {{ font-size: 12px; font-weight: 500; fill: #475569; }}
      .pill {{ font-size: 11.5px; font-weight: 700; fill: #ffffff; }}
    </style>
  </defs>

  <rect width="{width}" height="{height}" fill="#f8fafc" />
  <rect x="0" y="0" width="{width}" height="5" fill="#4338ca" />

  <!-- Header -->
  <g transform="translate(60, 45)">
    <text x="0" y="0" class="title">Fig. 4. Intermediate Feature-Level Fusion Pipeline and Dense Classification Head</text>
    <text x="0" y="26" class="sub">Vector Concatenation (192-d / 256-d), Deep Regularized Dense Compression, and 5-Syndrome Softmax Output</text>
  </g>

  <!-- 4 Modality Embeddings (Left Column) -->
  <!-- 1. Visual (128-d) -->
  <g filter="url(#shadow)">
    <rect x="60" y="110" width="260" height="95" rx="10" fill="#ffffff" stroke="#bae6fd" stroke-width="1.8"/>
    <rect x="60" y="110" width="6" height="95" rx="3" fill="#0284c7"/>
    <text x="80" y="140" class="box-t">Visual CNN Embedding</text>
    <text x="80" y="160" class="box-d">GeriatricNet / ResNet50</text>
    <rect x="80" y="172" width="120" height="22" rx="5" fill="#0284c7"/>
    <text x="140" y="187" text-anchor="middle" class="pill">Vector: 128-d</text>
  </g>

  <!-- 2. Audio (32-d) -->
  <g filter="url(#shadow)">
    <rect x="60" y="230" width="260" height="95" rx="10" fill="#ffffff" stroke="#fde68a" stroke-width="1.8"/>
    <rect x="60" y="230" width="6" height="95" rx="3" fill="#d97706"/>
    <text x="80" y="260" class="box-t">Audio CNN Embedding</text>
    <text x="80" y="280" class="box-d">Log Mel-Spectrogram (128×64)</text>
    <rect x="80" y="292" width="120" height="22" rx="5" fill="#d97706"/>
    <text x="140" y="307" text-anchor="middle" class="pill">Vector: 32-d</text>
  </g>

  <!-- 3. Clinical (32-d) -->
  <g filter="url(#shadow)">
    <rect x="60" y="350" width="260" height="95" rx="10" fill="#ffffff" stroke="#a7f3d0" stroke-width="1.8"/>
    <rect x="60" y="350" width="6" height="95" rx="3" fill="#059669"/>
    <text x="80" y="380" class="box-t">Clinical ANN Embedding</text>
    <text x="80" y="400" class="box-d">10 Standardized Biomarkers</text>
    <rect x="80" y="412" width="120" height="22" rx="5" fill="#059669"/>
    <text x="140" y="427" text-anchor="middle" class="pill">Vector: 32-d</text>
  </g>

  <!-- 4. Video (64-d) -->
  <g filter="url(#shadow)">
    <rect x="60" y="470" width="260" height="95" rx="10" fill="#ffffff" stroke="#ddd6fe" stroke-width="1.8"/>
    <rect x="60" y="470" width="6" height="95" rx="3" fill="#7c3aed"/>
    <text x="80" y="500" class="box-t">Video BiGRU Embedding</text>
    <text x="80" y="520" class="box-d">16 Frames Gait Dynamics</text>
    <rect x="80" y="532" width="120" height="22" rx="5" fill="#7c3aed"/>
    <text x="140" y="547" text-anchor="middle" class="pill">Vector: 64-d</text>
  </g>

  <!-- Connectors into Concatenate -->
  <path d="M 320 157 C 365 157, 365 340, 410 340" fill="none" stroke="#0284c7" stroke-width="2.2" marker-end="url(#arr)"/>
  <path d="M 320 277 C 365 277, 365 340, 410 340" fill="none" stroke="#d97706" stroke-width="2.2" marker-end="url(#arr)"/>
  <path d="M 320 397 C 365 397, 365 340, 410 340" fill="none" stroke="#059669" stroke-width="2.2" marker-end="url(#arr)"/>
  <path d="M 320 517 C 365 517, 365 340, 410 340" fill="none" stroke="#7c3aed" stroke-width="2.2" marker-end="url(#arr)"/>

  <!-- Column 2: Intermediate Concatenation Layer -->
  <g filter="url(#shadow)">
    <rect x="420" y="160" width="310" height="360" rx="14" fill="#ffffff" stroke="#c7d2fe" stroke-width="2"/>
    <rect x="420" y="160" width="8" height="360" rx="4" fill="#4338ca"/>
    <text x="445" y="195" font-size="16" font-weight="800" fill="#1e1b4b">Concatenation Layer</text>
    <text x="445" y="218" class="box-d">layers.Concatenate(axis=-1)</text>

    <!-- 3-Branch Box -->
    <rect x="445" y="240" width="260" height="100" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2"/>
    <text x="460" y="266" font-size="13" font-weight="700" fill="#0369a1">3-Branch Configuration:</text>
    <text x="460" y="292" font-size="20" font-weight="800" fill="#0284c7">192 Dimensions</text>
    <text x="460" y="320" font-size="11.5" font-weight="600" fill="#475569">128 (Img) + 32 (Aud) + 32 (Clin)</text>

    <!-- 4-Branch Box -->
    <rect x="445" y="360" width="260" height="110" rx="8" fill="#faf5ff" stroke="#c084fc" stroke-width="1.2"/>
    <text x="460" y="386" font-size="13" font-weight="700" fill="#6d28d9">4-Branch Configuration:</text>
    <text x="460" y="412" font-size="20" font-weight="800" fill="#7c3aed">256 Dimensions</text>
    <text x="460" y="438" font-size="11.5" font-weight="600" fill="#475569">+ 64 Video Temporal Embedding</text>
    <text x="460" y="456" font-size="11" font-weight="600" fill="#7c3aed">Acc: 95.10% (Quad-Modal)</text>
  </g>

  <!-- Arrow to Dense 1 -->
  <line x1="730" y1="340" x2="800" y2="340" stroke="#4338ca" stroke-width="2.5" marker-end="url(#arr)"/>

  <!-- Column 3: Dense Block 1 -->
  <g filter="url(#shadow)">
    <rect x="810" y="210" width="220" height="260" rx="12" fill="#ffffff" stroke="#c7d2fe" stroke-width="1.8"/>
    <rect x="810" y="210" width="6" height="260" rx="3" fill="#4f46e5"/>
    <text x="830" y="245" font-size="15" font-weight="800" fill="#0f172a">Dense Block 1</text>
    <text x="830" y="275" class="box-d">• Dense Layer (256)</text>
    <text x="830" y="300" class="box-d">• Batch Normalization</text>
    <text x="830" y="325" class="box-d">• ReLU Activation</text>
    <text x="830" y="350" class="box-d">• Dropout (rate = 0.40)</text>
    <text x="830" y="375" class="box-d">• L2 Reg (λ = 1e-4)</text>
    <rect x="830" y="410" width="180" height="26" rx="6" fill="#4f46e5"/>
    <text x="920" y="427" text-anchor="middle" class="pill">256 Features</text>
  </g>

  <!-- Arrow to Dense 2 -->
  <line x1="1030" y1="340" x2="1100" y2="340" stroke="#4338ca" stroke-width="2.5" marker-end="url(#arr)"/>

  <!-- Column 4: Dense Block 2 -->
  <g filter="url(#shadow)">
    <rect x="1110" y="210" width="220" height="260" rx="12" fill="#ffffff" stroke="#c7d2fe" stroke-width="1.8"/>
    <rect x="1110" y="210" width="6" height="260" rx="3" fill="#4f46e5"/>
    <text x="1130" y="245" font-size="15" font-weight="800" fill="#0f172a">Dense Block 2</text>
    <text x="1130" y="275" class="box-d">• Dense Layer (128)</text>
    <text x="1130" y="300" class="box-d">• Batch Normalization</text>
    <text x="1130" y="325" class="box-d">• ReLU Activation</text>
    <text x="1130" y="350" class="box-d">• Dropout (rate = 0.30)</text>
    <text x="1130" y="375" class="box-d">• Joint Latent Space</text>
    <rect x="1130" y="410" width="180" height="26" rx="6" fill="#4f46e5"/>
    <text x="1220" y="427" text-anchor="middle" class="pill">128-d Joint State</text>
  </g>

  <!-- Arrow to Softmax -->
  <line x1="1330" y1="340" x2="1390" y2="340" stroke="#e11d48" stroke-width="2.5" marker-end="url(#arr)"/>

  <!-- Column 5: Softmax Head -->
  <g filter="url(#shadow)">
    <rect x="1400" y="160" width="160" height="360" rx="12" fill="#ffffff" stroke="#fecdd3" stroke-width="2"/>
    <rect x="1400" y="160" width="6" height="360" rx="3" fill="#e11d48"/>
    <text x="1418" y="195" font-size="15" font-weight="800" fill="#9f1239">Softmax Head</text>
    <text x="1418" y="216" font-size="11.5" font-weight="600" fill="#be123c">Dense(5, 'softmax')</text>
    <line x1="1418" y1="228" x2="1542" y2="228" stroke="#f1f5f9" stroke-width="1.5"/>

    <text x="1418" y="255" class="box-t" font-size="12">1. Frailty</text>
    <text x="1418" y="295" class="box-t" font-size="12">2. Sarcopenia</text>
    <text x="1418" y="335" class="box-t" font-size="12">3. Cognitive Imp.</text>
    <text x="1418" y="375" class="box-t" font-size="12">4. Fall Risk</text>
    <text x="1418" y="415" class="box-t" font-size="12">5. Depression</text>

    <rect x="1415" y="460" width="130" height="26" rx="6" fill="#e11d48"/>
    <text x="1480" y="477" text-anchor="middle" class="pill">Σ p = 1.0</text>
  </g>

  <!-- Bottom Strip -->
  <rect x="60" y="610" width="1500" height="45" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.2" filter="url(#shadow)"/>
  <text x="80" y="638" font-size="12.5" font-weight="600" fill="#475569">Equations: V_fused = [v_img ‖ v_aud ‖ v_clin ‖ (v_vid)]  |  H1 = Dropout0.4(ReLU(BN(W1·V + b1)))  |  H2 = Dropout0.3(ReLU(BN(W2·H1 + b2)))  |  ŷ = Softmax(W3·H2 + b3)</text>
</svg>
''')
    return "".join(svg)

svg_f = create_fusion_svg()
with open("Figure_4_Feature_Fusion_Architecture.svg", "w", encoding="utf-8") as f:
    f.write(svg_f)
print("Created Figure_4_Feature_Fusion_Architecture.svg")
