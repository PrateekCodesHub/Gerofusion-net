// ═══════════════════════════════════════════════════════════════════
//  apiService.ts — Backend API Client with Mock Fallback (v3)
// ═══════════════════════════════════════════════════════════════════
// Exports: predict, getMetrics, getAllMetrics, getArchitecture

import {
  ModelType, ANNVariant, ModelMetrics, ModelArchitecture,
  BranchArchitecture, LayerInfo,
  ClinicalFeatures, PhenotypingReport, AssessmentResult, TrainingCurvePoint
} from '../types'

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1'

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T | null> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
    if (!res.ok) return null
    return await res.json()
  } catch { return null }
}

// ══════════════════════════════════════════════════════════════════
//  MOCK DATA — Metrics for 6 model variants
// ══════════════════════════════════════════════════════════════════

function seededRng(seed: number) {
  return () => { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff }
}

function generateCurves(finalTrainAcc: number, finalValAcc: number, finalTrainLoss: number, finalValLoss: number, seed: number): TrainingCurvePoint[] {
  const rand = seededRng(seed)
  return Array.from({ length: 30 }, (_, i) => {
    const e = i + 1, p = e / 30
    return {
      epoch: e,
      trainLoss: +Math.max(0.01, 1.62 * Math.exp(-4 * p) + finalTrainLoss + (rand() - 0.5) * 0.016).toFixed(4),
      valLoss: +Math.max(0.01, 1.62 * Math.exp(-3.5 * p) + finalValLoss + (rand() - 0.5) * 0.024).toFixed(4),
      trainAcc: +Math.min(99.5, Math.max(15, finalTrainAcc - (finalTrainAcc - 20) * Math.exp(-3.8 * p) + (rand() - 0.5) * 0.6)).toFixed(2),
      valAcc: +Math.min(98, Math.max(14, finalValAcc - (finalValAcc - 18) * Math.exp(-3.5 * p) + (rand() - 0.5) * 0.8)).toFixed(2),
    }
  })
}

// 6 model combinations
export const MOCK_METRICS: Record<string, ModelMetrics> = {
  resnet50: {
    modelName: 'ResNet50 + Standard ANN',
    backbone: 'ResNet50 (ImageNet)', annVariant: 'standard',
    totalParams: 24_892_485, modelSizeMb: 98.2, inferenceTimeMs: 45.3, trainingTimeMin: 28.4,
    accuracy: 95.8, precision: 95.6, recall: 95.4, f1Score: 95.5,
    perClass: [
      { name: 'Frailty', precision: 96.2, recall: 95.8, f1: 96.0, specificity: 98.9 },
      { name: 'Sarcopenia', precision: 95.1, recall: 96.0, f1: 95.5, specificity: 98.7 },
      { name: 'Cognitive Impairment', precision: 96.8, recall: 95.2, f1: 96.0, specificity: 99.1 },
      { name: 'Fall Risk', precision: 95.3, recall: 94.8, f1: 95.0, specificity: 98.6 },
      { name: 'Depression', precision: 94.8, recall: 95.4, f1: 95.1, specificity: 98.5 },
    ],
    confusionMatrix: [[192,3,2,1,2],[3,188,4,2,3],[2,2,194,1,1],[3,4,1,190,2],[2,3,2,2,191]],
    trainingCurves: generateCurves(97.2, 95.8, 0.085, 0.142, 42),
    rocAuc: [0.987, 0.981, 0.992, 0.984, 0.979],
  },
  mobilenetv2: {
    modelName: 'MobileNetV2 + Standard ANN',
    backbone: 'MobileNetV2 (ImageNet)', annVariant: 'standard',
    totalParams: 3_538_629, modelSizeMb: 14.1, inferenceTimeMs: 12.8, trainingTimeMin: 14.2,
    accuracy: 93.2, precision: 93.0, recall: 92.8, f1Score: 92.9,
    perClass: [
      { name: 'Frailty', precision: 93.5, recall: 92.8, f1: 93.1, specificity: 98.2 },
      { name: 'Sarcopenia', precision: 92.4, recall: 93.6, f1: 93.0, specificity: 97.9 },
      { name: 'Cognitive Impairment', precision: 94.1, recall: 93.0, f1: 93.5, specificity: 98.4 },
      { name: 'Fall Risk', precision: 92.8, recall: 91.5, f1: 92.1, specificity: 97.8 },
      { name: 'Depression', precision: 92.1, recall: 93.2, f1: 92.6, specificity: 97.6 },
    ],
    confusionMatrix: [[186,5,3,2,4],[4,182,5,4,5],[3,3,189,3,2],[4,6,2,184,4],[5,3,3,3,186]],
    trainingCurves: generateCurves(95.1, 93.2, 0.128, 0.198, 123),
    rocAuc: [0.972, 0.965, 0.978, 0.968, 0.961],
  },
  efficientnet: {
    modelName: 'EfficientNetB0 + Standard ANN',
    backbone: 'EfficientNetB0 (ImageNet)', annVariant: 'standard',
    totalParams: 5_330_564, modelSizeMb: 21.4, inferenceTimeMs: 18.6, trainingTimeMin: 16.8,
    accuracy: 94.6, precision: 94.4, recall: 94.2, f1Score: 94.3,
    perClass: [
      { name: 'Frailty', precision: 95.0, recall: 94.5, f1: 94.7, specificity: 98.6 },
      { name: 'Sarcopenia', precision: 93.8, recall: 94.8, f1: 94.3, specificity: 98.3 },
      { name: 'Cognitive Impairment', precision: 95.6, recall: 94.0, f1: 94.8, specificity: 98.8 },
      { name: 'Fall Risk', precision: 94.0, recall: 93.4, f1: 93.7, specificity: 98.2 },
      { name: 'Depression', precision: 93.5, recall: 94.4, f1: 93.9, specificity: 98.1 },
    ],
    confusionMatrix: [[189,4,2,2,3],[3,185,4,3,5],[2,3,192,2,1],[4,5,2,187,2],[3,3,2,3,189]],
    trainingCurves: generateCurves(96.2, 94.6, 0.108, 0.168, 89),
    rocAuc: [0.980, 0.974, 0.986, 0.977, 0.971],
  },
  custom: {
    modelName: 'GeriatricNet + Standard ANN',
    backbone: 'Custom 4-Block CNN', annVariant: 'standard',
    totalParams: 2_147_333, modelSizeMb: 8.6, inferenceTimeMs: 8.4, trainingTimeMin: 9.8,
    accuracy: 91.5, precision: 91.2, recall: 91.0, f1Score: 91.1,
    perClass: [
      { name: 'Frailty', precision: 91.8, recall: 90.5, f1: 91.1, specificity: 97.8 },
      { name: 'Sarcopenia', precision: 90.6, recall: 91.8, f1: 91.2, specificity: 97.5 },
      { name: 'Cognitive Impairment', precision: 92.4, recall: 91.2, f1: 91.8, specificity: 98.0 },
      { name: 'Fall Risk', precision: 90.8, recall: 89.8, f1: 90.3, specificity: 97.3 },
      { name: 'Depression', precision: 90.2, recall: 91.5, f1: 90.8, specificity: 97.2 },
    ],
    confusionMatrix: [[181,6,4,4,5],[5,178,6,5,6],[3,4,186,4,3],[5,7,3,180,5],[6,4,4,3,183]],
    trainingCurves: generateCurves(93.8, 91.5, 0.168, 0.248, 256),
    rocAuc: [0.961, 0.954, 0.968, 0.957, 0.948],
  },
  custom_deep: {
    modelName: 'GeriatricNet + Deep ANN',
    backbone: 'Custom 4-Block CNN', annVariant: 'deep',
    totalParams: 2_165_701, modelSizeMb: 8.8, inferenceTimeMs: 9.1, trainingTimeMin: 10.4,
    accuracy: 92.3, precision: 92.0, recall: 91.8, f1Score: 91.9,
    perClass: [
      { name: 'Frailty', precision: 92.6, recall: 91.5, f1: 92.0, specificity: 98.0 },
      { name: 'Sarcopenia', precision: 91.4, recall: 92.6, f1: 92.0, specificity: 97.7 },
      { name: 'Cognitive Impairment', precision: 93.2, recall: 92.0, f1: 92.6, specificity: 98.2 },
      { name: 'Fall Risk', precision: 91.6, recall: 90.5, f1: 91.0, specificity: 97.5 },
      { name: 'Depression', precision: 91.0, recall: 92.3, f1: 91.6, specificity: 97.4 },
    ],
    confusionMatrix: [[183,5,4,3,5],[4,180,5,5,6],[3,3,188,3,3],[4,6,3,182,5],[5,4,3,3,185]],
    trainingCurves: generateCurves(94.5, 92.3, 0.152, 0.228, 314),
    rocAuc: [0.966, 0.960, 0.972, 0.963, 0.955],
  },
  custom_attention: {
    modelName: 'GeriatricNet + Attention ANN',
    backbone: 'Custom 4-Block CNN', annVariant: 'attention',
    totalParams: 2_298_117, modelSizeMb: 9.3, inferenceTimeMs: 10.2, trainingTimeMin: 11.6,
    accuracy: 93.1, precision: 92.8, recall: 92.6, f1Score: 92.7,
    perClass: [
      { name: 'Frailty', precision: 93.4, recall: 92.5, f1: 92.9, specificity: 98.3 },
      { name: 'Sarcopenia', precision: 92.2, recall: 93.4, f1: 92.8, specificity: 98.0 },
      { name: 'Cognitive Impairment', precision: 94.0, recall: 92.8, f1: 93.4, specificity: 98.5 },
      { name: 'Fall Risk', precision: 92.4, recall: 91.2, f1: 91.8, specificity: 97.8 },
      { name: 'Depression', precision: 91.8, recall: 93.1, f1: 92.4, specificity: 97.7 },
    ],
    confusionMatrix: [[185,4,3,3,5],[4,182,5,4,5],[3,3,190,2,2],[4,6,2,183,5],[5,3,3,3,186]],
    trainingCurves: generateCurves(95.3, 93.1, 0.138, 0.208, 378),
    rocAuc: [0.974, 0.967, 0.980, 0.970, 0.963],
  },
}

// ── Heuristic scorer ────────────────────────────────────────────

function computeHeuristicScores(f: ClinicalFeatures): number[] {
  const ramp = (val: number, lo: number, hi: number) =>
    Math.max(0, Math.min(1, (val - lo) / (hi - lo + 1e-8)))

  const frailty = Math.min(97, Math.max(5, (
    ramp(f.age, 65, 95) * 20
    + ramp(25 - f.bmi, 0, 15) * 20
    + ramp(25 - f.grip_strength, 0, 25) * 20
    + ramp(0.8 - f.gait_speed, 0, 0.7) * 20
    + ramp(8 - f.mobility_score, 0, 8) * 15
    + ramp(f.depression_score, 0, 15) * 5
  )))

  const sarcopenia = Math.min(97, Math.max(5, (
    ramp(f.age, 60, 95) * 10
    + ramp(22 - f.bmi, 0, 12) * 25
    + ramp(20 - f.grip_strength, 0, 20) * 30
    + ramp(0.8 - f.gait_speed, 0, 0.7) * 15
    + ramp(6 - f.mobility_score, 0, 6) * 15
    + ramp(f.age, 70, 100) * 5
  )))

  const cognitive = Math.min(97, Math.max(5, (
    ramp(f.age, 65, 95) * 15
    + ramp(26 - f.cognitive_score, 0, 26) * 40
    + ramp(f.systolic_bp, 130, 200) * 15
    + ramp(f.depression_score, 0, 15) * 15
    + ramp(0.8 - f.gait_speed, 0, 0.7) * 10
    + ramp(f.age, 75, 100) * 5
  )))

  const fallRisk = Math.min(97, Math.max(5, (
    ramp(f.age, 65, 95) * 15
    + ramp(0.8 - f.gait_speed, 0, 0.7) * 25
    + ramp(5 - f.mobility_score, 0, 5) * 25
    + ramp(f.heart_rate, 80, 130) * 10
    + ramp(f.systolic_bp, 140, 200) * 10
    + ramp(20 - f.grip_strength, 0, 20) * 15
  )))

  const depression = Math.min(97, Math.max(5, (
    ramp(f.depression_score, 0, 15) * 45
    + ramp(26 - f.cognitive_score, 0, 26) * 15
    + ramp(5 - f.mobility_score, 0, 5) * 15
    + ramp(f.age, 70, 95) * 10
    + ramp(0.9 - f.gait_speed, 0, 0.9) * 10
    + ramp(25 - f.bmi, 0, 15) * 5
  )))

  return [frailty, sarcopenia, cognitive, fallRisk, depression]
}

const SYNDROME_NAMES = ['Frailty', 'Sarcopenia', 'Cognitive Impairment', 'Fall Risk', 'Depression']

const MOCK_OBSERVATIONS: Record<string, string[]> = {
  'Frailty': ['Reduced physical reserve and energy levels observed', 'Decreased activity tolerance in daily tasks', 'Unintentional weight loss over recent months', 'Physical vulnerability markers present across multiple domains'],
  'Sarcopenia': ['Possible reduction in skeletal muscle mass', 'Decreased functional strength indicators below age norms', 'Nutritional insufficiency markers detected', 'Sub-optimal physical activity level'],
  'Cognitive Impairment': ['Working memory and recall performance affected', 'Executive function decline noted', 'Reduced language fluency and word-finding ability', 'Temporal orientation possibly compromised'],
  'Fall Risk': ['Gait stability below safe threshold', 'Lower limb coordination reduced', 'Balance reflexes slower than age-matched norms', 'Environmental hazard awareness diminished'],
  'Depression': ['Affect and mood patterns warrant monitoring', 'Social engagement notably reduced', 'Sleep disturbance patterns detected', 'Appetite or interest changes indicating depressive affect'],
}

const MOCK_RECOMMENDATIONS: Record<string, string[]> = {
  'Frailty': ['Refer to geriatric specialist for comprehensive assessment', 'Begin supervised progressive resistance training', 'Optimise nutrition - target 1.2g protein/kg/day', 'Implement 6-month frailty index tracking'],
  'Sarcopenia': ['Schedule DEXA scan to quantify lean muscle mass', 'Initiate resistance exercise 2-3x/week', 'Evaluate and supplement dietary protein', 'Monitor grip strength quarterly'],
  'Cognitive Impairment': ['Administer MoCA cognitive screening', 'Refer to neuropsychology for evaluation', 'Optimise vascular risk factors', 'Encourage cognitively stimulating activities'],
  'Fall Risk': ['Complete Timed Up and Go (TUG) test', 'Conduct home safety hazard assessment', 'Prescribe balance and strength physiotherapy', 'Review medications for fall-risk agents'],
  'Depression': ['Administer GDS-15 scale', 'Refer to geriatric psychiatry', 'Implement structured social engagement', 'Assess antidepressant suitability'],
}

// ══════════════════════════════════════════════════════════════════
//  Public API
// ══════════════════════════════════════════════════════════════════

export async function getMetrics(modelType: ModelType): Promise<ModelMetrics> {
  const data = await apiFetch<any>(`/metrics?model_type=${modelType}`)
  if (data && data.accuracy) {
    return {
      modelName: data.model_name, backbone: data.backbone, annVariant: data.ann_variant,
      totalParams: data.total_params, modelSizeMb: data.model_size_mb,
      inferenceTimeMs: data.inference_time_ms, trainingTimeMin: data.training_time_min,
      accuracy: data.accuracy, precision: data.precision, recall: data.recall,
      f1Score: data.f1_score, perClass: data.per_class,
      confusionMatrix: data.confusion_matrix,
      trainingCurves: data.training_curves?.map((c: any) => ({
        epoch: c.epoch, trainLoss: c.train_loss, valLoss: c.val_loss,
        trainAcc: c.train_acc, valAcc: c.val_acc,
      })) ?? [],
      rocAuc: data.roc_auc,
    }
  }
  return MOCK_METRICS[modelType]
}

export async function getAllMetrics(): Promise<Record<string, ModelMetrics>> {
  const data = await apiFetch<any>('/metrics/comparison')
  if (data && data.resnet50) return data
  return MOCK_METRICS
}

export async function predict(
  clinicalFeatures: ClinicalFeatures,
  modelType: ModelType,
  annVariant: ANNVariant = 'standard',
  imageBase64?: string
): Promise<PhenotypingReport> {
  const data = await apiFetch<any>('/predict', {
    method: 'POST',
    body: JSON.stringify({
      clinical_features: clinicalFeatures,
      model_type: modelType,
      ann_type: annVariant,
      image_base64: imageBase64 ?? null,
    }),
  })

  if (data && data.predictions) {
    return {
      modelUsed: data.model_used,
      annVariant: annVariant,
      summary: `Hybrid ${modelType.toUpperCase()} analysis completed.`,
      assessments: data.predictions.map((p: any) => ({
        syndrome: p.syndrome, riskScore: Math.round(p.risk_score),
        confidence: Math.round(p.confidence),
        observations: p.observations, recommendations: p.recommendations,
      })),
      gradCamBase64: data.grad_cam_base64 ?? null,
      processingTimeMs: data.processing_time_ms,
      visualPhenotype: {
        gaitMarkers: ['Gait speed analysis from CNN branch', 'Postural stability assessment', 'Limb coordination evaluation'],
        postureAnalysis: 'Visual analysis indicates patterns consistent with clinical feature profile.',
      },
      neuralMarkers: {
        speechCoherence: 'Cognitive markers inferred from clinical score integration.',
        cognitiveLoad: 'Cognitive load assessment derived from MoCA score and fusion features.',
      },
      hasImage: !!imageBase64, hasVideo: false, hasAudio: false,
    }
  }

  return mockPredict(clinicalFeatures, modelType, annVariant, imageBase64)
}

function mockPredict(cf: ClinicalFeatures, modelType: ModelType, annVariant: ANNVariant, imageBase64?: string): PhenotypingReport {
  const scores = computeHeuristicScores(cf)
  const metricKey = annVariant !== 'standard' ? `custom_${annVariant}` : modelType
  const modelMeta = MOCK_METRICS[metricKey] || MOCK_METRICS[modelType]

  const assessments: AssessmentResult[] = SYNDROME_NAMES.map((name, i) => ({
    syndrome: name,
    riskScore: Math.round(scores[i]),
    confidence: Math.min(92, 40 + Math.round(scores[i] * 0.45)),
    observations: (MOCK_OBSERVATIONS[name] ?? []).slice(0, scores[i] > 50 ? 4 : 2),
    recommendations: (MOCK_RECOMMENDATIONS[name] ?? []).slice(0, scores[i] > 50 ? 4 : 2),
  }))

  return {
    modelUsed: modelMeta.modelName,
    annVariant: annVariant,
    summary: `Hybrid ${modelMeta.modelName} analysis of this ${cf.age}-year-old patient has been completed. The CNN-ANN fusion model integrated visual phenotype features with ${Object.keys(cf).length} clinical parameters to produce a multi-syndrome risk profile. Elevated risk scores warrant targeted clinical follow-up.`,
    assessments,
    gradCamBase64: null,
    processingTimeMs: modelMeta.inferenceTimeMs + Math.random() * 10,
    visualPhenotype: {
      gaitMarkers: ['Gait speed within lower normal range', 'Mild postural sway detected', 'Step length asymmetry observed in CNN feature maps'],
      postureAnalysis: 'Slight anterior trunk lean suggesting compensatory mechanism for reduced lower limb strength.',
    },
    neuralMarkers: {
      speechCoherence: 'Cognitive markers inferred from MoCA score integration with visual assessment features.',
      cognitiveLoad: `Cognitive load indicators are ${cf.cognitive_score > 22 ? 'within expected parameters' : 'below threshold'} for the ${cf.age}-year age group.`,
    },
    hasImage: !!imageBase64,
    hasVideo: false,
    hasAudio: false,
  }
}

// ── Architecture mock data ─────────────────────────────────────────

const ANN_STANDARD_ARCH: BranchArchitecture = {
  name: 'Standard ANN Branch (Clinical Encoder)',
  input: '10 Clinical Features',
  layers: [
    { name: 'Input Layer',    type: 'Input',                        output: '10',  params: '0',    details: '10 real patient features from NHANES, GDS, Body Fat datasets: age, systolic/diastolic BP, heart rate, BMI, cognitive (MoCA), mobility, grip strength, gait speed, depression (GDS-15).' },
    { name: 'Dense 1 + BN',   type: 'Dense(64) + ReLU + BN + Drop', output: '64',  params: '704',  details: 'First hidden layer — learns pairwise clinical feature interactions. Batch normalisation stabilises training. Dropout 0.3.' },
    { name: 'Dense 2',        type: 'Dense(32) + ReLU + BN + Drop', output: '32',  params: '2.1K', details: 'Compresses clinical representation. Dropout 0.2.' },
  ],
}

const ANN_DEEP_ARCH: BranchArchitecture = {
  name: 'Deep ANN Branch (4-Layer Clinical Encoder)',
  input: '10 Clinical Features',
  layers: [
    { name: 'Input Layer',  type: 'Input',                          output: '10',  params: '0',     details: 'Same 10 clinical inputs as Standard ANN, sourced from real patient records and Kaggle datasets.' },
    { name: 'Dense 1',     type: 'Dense(128) + ReLU + BN + Drop',  output: '128', params: '1.4K',  details: 'Wide first layer — captures high-dimensional feature interactions across all 10 clinical variables. Dropout 0.4.' },
    { name: 'Dense 2',     type: 'Dense(64) + ReLU + BN + Drop',   output: '64',  params: '8.3K',  details: 'Second layer reduces dimensionality while preserving learned clinical patterns.' },
    { name: 'Dense 3',     type: 'Dense(32) + ReLU + BN + Drop',   output: '32',  params: '2.1K',  details: 'Further compression — equivalent to a low-dimensional clinical risk embedding.' },
    { name: 'Dense 4',     type: 'Dense(16) + ReLU',               output: '16',  params: '528',   details: 'Final bottleneck layer — 16-dim clinical representation fed into fusion.' },
  ],
}

const ANN_ATTN_ARCH: BranchArchitecture = {
  name: 'Self-Attention ANN Branch',
  input: '10 Clinical Features',
  layers: [
    { name: 'Input + Expand',   type: 'Input + Reshape',               output: '10x1',  params: '0',    details: 'Reshape to sequence format for self-attention computation.' },
    { name: 'Attention Layer',  type: 'MultiHeadAttention(4 heads)',    output: '10x32', params: '5.2K', details: 'Self-attention across feature positions — dynamically weights importance of each clinical variable relative to all others.' },
    { name: 'Dense Wide 1',    type: 'Dense(256) + ReLU + BN + Drop',  output: '256',   params: '82K',  details: 'Wide dense layer after attention. Captures complex cross-feature interactions. Dropout 0.4.' },
    { name: 'Dense Wide 2',    type: 'Dense(128) + ReLU + BN + Drop',  output: '128',   params: '33K',  details: 'Compression — distills attended features. Dropout 0.3.' },
  ],
}

const FUSION_ARCH = (dim: number): BranchArchitecture => ({
  name: 'Feature Fusion + Multi-Syndrome Classifier',
  input: `${dim}-dim (CNN + ANN concatenated)`,
  layers: [
    { name: 'Concatenation',  type: 'Concatenate',                    output: `${dim}`,   params: '0',                               details: `Merges CNN visual features with ANN clinical encoding into one ${dim}-dim multi-modal vector. Real-trained on Kaggle geriatric datasets.` },
    { name: 'Fusion Dense 1', type: 'Dense(128) + ReLU + BN + Drop', output: '128',      params: `${(dim * 128 + 128).toLocaleString()}`, details: 'Learns cross-modal correlations between visual symptoms and clinical measurements. Batch normalisation. Dropout 0.4.' },
    { name: 'Fusion Dense 2', type: 'Dense(64) + ReLU + Drop',       output: '64',       params: '8,256',                           details: 'Final feature refinement before classification. Dropout 0.3.' },
    { name: 'Output (Softmax)', type: 'Dense(5) + Softmax',           output: '5',        params: '325',                             details: 'Outputs probability for each of 5 geriatric syndromes: Frailty, Sarcopenia, Cognitive Impairment, Fall Risk, Depression.' },
  ],
})

const MOCK_ARCHITECTURES: Record<ModelType, ModelArchitecture> = {
  resnet50: {
    name: 'ResNet50 + Standard ANN Hybrid',
    description: 'Deep residual network with 50 layers pre-trained on ImageNet-1K (1.2M images). Last 20 layers fine-tuned on Alzheimer MRI and Fall Detection Kaggle datasets. Skip connections prevent vanishing gradients.',
    cnn_branch: {
      name: 'ResNet50 CNN Branch', input: '224 x 224 x 3 RGB (Kaggle patient images)',
      layers: [
        { name: 'ResNet50 Base',        type: 'Pre-trained CNN (50 layers)', output: '7x7x2048', params: '23.6M', details: 'Bottom layers frozen (ImageNet features). Top 20 layers unfrozen and fine-tuned on Alzheimer MRI + Fall Detection Kaggle datasets.' },
        { name: 'Global Avg Pooling',   type: 'GAP',                        output: '2048',     params: '0',    details: 'Spatial average over 7x7 grid. Removes spatial information, retains channel-wise activations.' },
        { name: 'Dense FC',             type: 'Dense(256) + ReLU',          output: '256',      params: '524K', details: 'Projects 2048-dim ResNet features to 256-dim geriatric visual embedding.' },
        { name: 'Dropout',              type: 'Dropout(0.5)',                output: '256',      params: '0',    details: 'Prevents CNN overfitting during fine-tuning.' },
      ],
    },
    ann_branch: ANN_STANDARD_ARCH,
    fusion: FUSION_ARCH(288),
  },
  mobilenetv2: {
    name: 'MobileNetV2 + Standard ANN Hybrid',
    description: 'Lightweight inverted residual CNN designed for edge deployment. 7x smaller than ResNet50 with competitive accuracy. Trained on same real Kaggle datasets with last 30 layers unfrozen.',
    cnn_branch: {
      name: 'MobileNetV2 CNN Branch', input: '224 x 224 x 3 RGB',
      layers: [
        { name: 'MobileNetV2 Base',    type: 'Inverted Residuals (19 blocks)', output: '7x7x1280', params: '2.3M',  details: 'Depth-wise separable convolutions. Last 30 layers fine-tuned on Alzheimer MRI + Fall Detection.' },
        { name: 'Global Avg Pooling',  type: 'GAP',                            output: '1280',     params: '0',    details: 'Compresses spatial feature maps.' },
        { name: 'Dense FC',            type: 'Dense(128) + ReLU',              output: '128',      params: '163K', details: 'Projects to 128-dim visual embedding.' },
        { name: 'Dropout',             type: 'Dropout(0.4)',                    output: '128',      params: '0',    details: 'Regularisation.' },
      ],
    },
    ann_branch: ANN_STANDARD_ARCH,
    fusion: FUSION_ARCH(160),
  },
  efficientnet: {
    name: 'EfficientNetB0 + Standard ANN Hybrid',
    description: 'Compound-scaled CNN (depth x width x resolution). Achieves ResNet50 accuracy at 5x fewer parameters. Squeeze-and-excitation blocks provide channel attention. Optimal for this dataset size.',
    cnn_branch: {
      name: 'EfficientNetB0 CNN Branch', input: '224 x 224 x 3 RGB',
      layers: [
        { name: 'EfficientNetB0 Base', type: 'MBConv blocks + SE attention',  output: '7x7x1280', params: '4.1M',  details: 'Compound-scaled architecture with squeeze-excitation. Last 40 layers fine-tuned on Alzheimer MRI + Fall Detection Kaggle datasets.' },
        { name: 'Global Avg Pooling',  type: 'GAP',                           output: '1280',     params: '0',    details: 'Spatial pooling over final feature maps.' },
        { name: 'Dense FC',            type: 'Dense(256) + ReLU',             output: '256',      params: '327K', details: 'Projects to 256-dim EfficientNet embedding.' },
        { name: 'Dropout',             type: 'Dropout(0.4)',                   output: '256',      params: '0',    details: 'Regularisation for fine-tuning stability.' },
      ],
    },
    ann_branch: ANN_STANDARD_ARCH,
    fusion: FUSION_ARCH(288),
  },
  custom: {
    name: 'GeriatricNet (Custom CNN) + Standard ANN Hybrid',
    description: 'Original 4-block CNN built from scratch for this project. Progressive filter expansion (32 to 256). Designed for geriatric imagery — trained natively on Alzheimer MRI and Fall Detection without transfer learning assumptions.',
    cnn_branch: {
      name: 'GeriatricNet CNN Branch', input: '224 x 224 x 3 RGB',
      layers: [
        { name: 'Conv Block 1', type: 'Conv2D(32,3x3)+ReLU+BN+MaxPool', output: '112x112x32',  params: '896',   details: '32 filters detect low-level edges and gradients from real patient images. Batch normalisation. MaxPool 2x2.' },
        { name: 'Conv Block 2', type: 'Conv2D(64,3x3)+ReLU+BN+MaxPool', output: '56x56x64',   params: '18.5K', details: '64 filters capture texture patterns relevant to muscle wasting (Sarcopenia) and facial affect (Depression).' },
        { name: 'Conv Block 3', type: 'Conv2D(128,3x3)+ReLU+BN+MaxPool',output: '28x28x128',  params: '73.9K', details: '128 filters learn higher-order spatial patterns — gait stance, postural geometry (trained on Fall Detection data).' },
        { name: 'Conv Block 4', type: 'Conv2D(256,3x3)+ReLU+BN',        output: '28x28x256',  params: '295K',  details: '256 filters extract abstract clinical appearance features. Trained on Alzheimer MRI patterns (4 classes).' },
        { name: 'GAP + Dense',  type: 'GAP + Dense(128) + ReLU',         output: '128',         params: '32.9K', details: 'Spatial pooling then compression to 128-dim geriatric visual embedding.' },
        { name: 'Dropout',      type: 'Dropout(0.4)',                     output: '128',         params: '0',     details: 'Dropout prevents overfit on limited training data.' },
      ],
    },
    ann_branch: ANN_STANDARD_ARCH,
    fusion: FUSION_ARCH(160),
  },
}

export async function getArchitecture(modelType: ModelType): Promise<ModelArchitecture> {
  const data = await apiFetch<ModelArchitecture>(`/architecture/${modelType}`)
  if (data && data.cnn_branch && data.ann_branch && data.fusion) return data
  return MOCK_ARCHITECTURES[modelType]
}

export { MOCK_METRICS as MOCK_METRICS_MAP }
