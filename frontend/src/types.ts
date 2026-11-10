// ═══════════════════════════════════════════════════════════════════
//  types.ts — Type Definitions for GeroFusion v3 Multi-Modal System
// ═══════════════════════════════════════════════════════════════════

// ── Geriatric Syndrome Classes ──────────────────────────────────────

export enum GeriatricSyndrome {
  FRAILTY = 'Frailty',
  SARCOPENIA = 'Sarcopenia',
  COGNITIVE_IMPAIRMENT = 'Cognitive Impairment',
  FALL_RISK = 'Fall Risk',
  DEPRESSION = 'Depression'
}

// ── Model Selection ─────────────────────────────────────────────────

export type ModelType = 'resnet50' | 'mobilenetv2' | 'efficientnet' | 'custom'
export type ANNVariant = 'standard' | 'deep' | 'attention'

export const MODEL_DISPLAY_NAMES: Record<ModelType, string> = {
  resnet50: 'ResNet50 Hybrid',
  mobilenetv2: 'MobileNetV2 Hybrid',
  efficientnet: 'EfficientNetB0 Hybrid',
  custom: 'GeriatricNet (Custom)'
}

export const MODEL_DESCRIPTIONS: Record<ModelType, string> = {
  resnet50: 'Deep residual network with skip connections. ImageNet pre-trained. Top 20 layers fine-tuned. Highest accuracy, largest model.',
  mobilenetv2: 'Lightweight depth-wise separable convolutions. 7x smaller than ResNet50. Ideal for mobile deployment.',
  efficientnet: 'Compound-scaled architecture (depth x width x resolution). Optimal accuracy-efficiency balance. Squeeze-and-excitation blocks.',
  custom: 'Original 4-block CNN designed for this project. Progressive filter expansion (32->64->128->256). Balanced accuracy and size.'
}

export const ANN_DISPLAY_NAMES: Record<ANNVariant, string> = {
  standard: 'Standard ANN (64→32)',
  deep: 'Deep ANN (128→64→32→16)',
  attention: 'Attention ANN (256→128)'
}

export const ANN_DESCRIPTIONS: Record<ANNVariant, string> = {
  standard: '2-layer baseline encoder. Learns pairwise clinical feature interactions.',
  deep: '4-layer deep encoder. Captures higher-order feature interactions (e.g., age × grip × gait).',
  attention: 'Wide layers with self-attention mechanism. Learns feature importance weights dynamically.'
}

// ── Clinical Features Input ─────────────────────────────────────────

export interface ClinicalFeatures {
  age: number
  systolic_bp: number
  diastolic_bp: number
  heart_rate: number
  bmi: number
  cognitive_score: number
  mobility_score: number
  grip_strength: number
  gait_speed: number
  depression_score: number
}

export const DEFAULT_CLINICAL_FEATURES: ClinicalFeatures = {
  age: 76,
  systolic_bp: 138,
  diastolic_bp: 74,
  heart_rate: 78,
  bmi: 23.5,
  cognitive_score: 22,
  mobility_score: 4.5,
  grip_strength: 18,
  gait_speed: 0.65,
  depression_score: 6
}

export const FEATURE_LABELS: Record<keyof ClinicalFeatures, { label: string; unit: string; min: number; max: number; step: number }> = {
  age:              { label: 'Age',                unit: 'years', min: 60, max: 100, step: 1 },
  systolic_bp:      { label: 'Systolic BP',        unit: 'mmHg',  min: 90, max: 200, step: 1 },
  diastolic_bp:     { label: 'Diastolic BP',       unit: 'mmHg',  min: 50, max: 120, step: 1 },
  heart_rate:       { label: 'Heart Rate',         unit: 'bpm',   min: 45, max: 130, step: 1 },
  bmi:              { label: 'BMI',                unit: 'kg/m²', min: 14, max: 45,  step: 0.1 },
  cognitive_score:  { label: 'Cognitive (MoCA)',    unit: '/30',   min: 0,  max: 30,  step: 1 },
  mobility_score:   { label: 'Mobility Score',     unit: '/10',   min: 0,  max: 10,  step: 0.5 },
  grip_strength:    { label: 'Grip Strength',      unit: 'kg',    min: 3,  max: 55,  step: 0.5 },
  gait_speed:       { label: 'Gait Speed',         unit: 'm/s',   min: 0.1, max: 1.8, step: 0.05 },
  depression_score: { label: 'Depression (GDS-15)', unit: '/15',   min: 0,  max: 15,  step: 1 },
}

// ── Prediction / Assessment Results ─────────────────────────────────

export interface AssessmentResult {
  syndrome: string
  riskScore: number
  confidence: number
  observations: string[]
  recommendations: string[]
}

export interface PhenotypingReport {
  modelUsed: string
  annVariant: string
  summary: string
  assessments: AssessmentResult[]
  gradCamBase64: string | null
  processingTimeMs: number
  visualPhenotype: { gaitMarkers: string[]; postureAnalysis: string }
  neuralMarkers: { speechCoherence: string; cognitiveLoad: string }
  hasImage: boolean
  hasVideo: boolean
  hasAudio: boolean
}

// ── Model Performance Metrics ───────────────────────────────────────

export interface PerClassMetric {
  name: string
  precision: number
  recall: number
  f1: number
  specificity: number
}

export interface TrainingCurvePoint {
  epoch: number
  trainLoss: number
  valLoss: number
  trainAcc: number
  valAcc: number
}

export interface ModelMetrics {
  modelName: string
  backbone: string
  annVariant?: string
  totalParams: number
  modelSizeMb: number
  inferenceTimeMs: number
  trainingTimeMin: number
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  perClass: PerClassMetric[]
  confusionMatrix: number[][]
  trainingCurves: TrainingCurvePoint[]
  rocAuc: number[]
}

// ── Architecture Layer Info ─────────────────────────────────────────

export interface LayerInfo {
  name: string
  type: string
  output: string
  params: string
  details: string
}

export interface BranchArchitecture {
  name: string
  input: string
  layers: LayerInfo[]
}

export interface ModelArchitecture {
  name: string
  description: string
  cnn_branch: BranchArchitecture
  ann_branch: BranchArchitecture
  fusion: BranchArchitecture
}

// ── Multi-modal Input ───────────────────────────────────────────────

export interface MultiModalInput {
  imageBase64?: string
  videoFramesBase64?: string[]
  audioBase64?: string
  clinicalFeatures: ClinicalFeatures
  modelType: ModelType
  annVariant: ANNVariant
}

// ── Navigation ──────────────────────────────────────────────────────

export type PageView = 'home' | 'analysis' | 'dashboard' | 'comparison' | 'architecture' | 'system'
