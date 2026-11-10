import React, { useState, useEffect } from 'react'
import { Layers, ChevronDown, ChevronRight, Cpu, GitBranch, ArrowRight } from 'lucide-react'
import { ModelType, MODEL_DISPLAY_NAMES, ModelArchitecture, LayerInfo, BranchArchitecture } from '../types'
import { getArchitecture } from '../services/apiService'

function LayerCard({ layer, index, color }: { layer: LayerInfo; index: number; color: string }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="bg-white rounded-xl border border-slate-100 shadow-sm overflow-hidden hover:shadow-md transition-all">
      <button onClick={() => setExpanded(!expanded)} className="w-full p-4 flex items-center gap-3 text-left">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white text-xs font-bold flex-shrink-0" style={{ background: color }}>
          {index + 1}
        </div>
        <div className="flex-grow min-w-0">
          <p className="font-bold text-slate-800 text-sm">{layer.name}</p>
          <p className="text-xs text-slate-400 truncate">{layer.type}</p>
        </div>
        <div className="text-right flex-shrink-0 mr-2">
          <p className="text-xs font-mono font-bold text-slate-600">{layer.output}</p>
          <p className="text-xs text-slate-400">{layer.params} params</p>
        </div>
        {expanded ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
      </button>
      {expanded && (
        <div className="px-4 pb-4 border-t border-slate-50 pt-3">
          <p className="text-sm text-slate-600 leading-relaxed">{layer.details}</p>
        </div>
      )}
    </div>
  )
}

function BranchSection({ branch, color, icon }: { branch: BranchArchitecture; color: string; icon: React.ReactNode }) {
  return (
    <div className="bg-slate-50 rounded-2xl p-5">
      <div className="flex items-center gap-2 mb-4">
        {icon}
        <div>
          <h3 className="font-bold text-slate-800">{branch.name}</h3>
          <p className="text-xs text-slate-400">Input: {branch.input}</p>
        </div>
      </div>
      <div className="space-y-2">
        {branch.layers.map((layer, i) => (
          <React.Fragment key={i}>
            <LayerCard layer={layer} index={i} color={color} />
            {i < branch.layers.length - 1 && (
              <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-slate-300 rotate-90" /></div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}

export default function ArchitectureExplorer() {
  const [selectedModel, setSelectedModel] = useState<ModelType>('resnet50')
  const [arch, setArch] = useState<ModelArchitecture | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    setArch(null)
    getArchitecture(selectedModel)
      .then((data) => {
        setArch(data)
        setLoading(false)
      })
      .catch(() => {
        setLoading(false)
      })
  }, [selectedModel])

  if (loading) {
    return (
      <div className="py-20 text-center">
        <div className="inline-block w-8 h-8 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mb-4"></div>
        <p className="text-slate-400">Loading architecture...</p>
      </div>
    )
  }

  if (!arch) {
    return (
      <div className="py-20 text-center">
        <p className="text-red-400 font-semibold">Failed to load architecture data.</p>
        <button
          onClick={() => setSelectedModel(selectedModel)}
          className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm"
        >
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto py-12 px-4 space-y-8">
      {/* Header */}
      <div className="text-center">
        <span className="inline-block bg-purple-50 text-purple-700 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider mb-3">Layer-by-Layer</span>
        <h1 className="text-4xl font-extrabold text-slate-900 mb-3">Architecture Explorer</h1>
        <p className="text-slate-500 max-w-xl mx-auto">{arch.description}</p>
      </div>

      {/* Model selector */}
      <div className="flex flex-wrap justify-center gap-2">
        {(['resnet50', 'mobilenetv2', 'efficientnet', 'custom'] as ModelType[]).map(m => (
          <button key={m} onClick={() => setSelectedModel(m)}
            className={`px-4 py-2.5 rounded-xl text-sm font-semibold transition-all ${selectedModel === m ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-200' : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-50'}`}>
            {MODEL_DISPLAY_NAMES[m]}
          </button>
        ))}
      </div>

      {/* Architecture overview card */}
      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
        <h2 className="text-lg font-bold text-slate-800 mb-1 flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-600" /> {arch.name}
        </h2>
        <p className="text-xs text-slate-400 mb-4">Training Data: Real Kaggle datasets — Alzheimer MRI, Fall Detection, NHANES, GDS, Body Fat</p>

        {/* Visual flow diagram — 4 modalities */}
        <div className="flex flex-col lg:flex-row gap-3 items-stretch mb-4">
          <div className="flex-1 bg-blue-50 rounded-xl p-3 border border-blue-100 text-center">
            <Cpu className="w-7 h-7 text-blue-600 mx-auto mb-1" />
            <p className="font-bold text-blue-800 text-xs">CNN Branch</p>
            <p className="text-xs text-blue-500">224×224×3 Image</p>
            <p className="text-xs text-blue-400">Alzheimer MRI / Fall Frames</p>
          </div>
          <div className="flex-1 bg-violet-50 rounded-xl p-3 border border-violet-100 text-center">
            <ArrowRight className="w-7 h-7 text-violet-600 mx-auto mb-1" />
            <p className="font-bold text-violet-800 text-xs">Video GRU Branch</p>
            <p className="text-xs text-violet-500">16 Frames × 224×224</p>
            <p className="text-xs text-violet-400">Gait / Posture Sequences</p>
          </div>
          <div className="flex items-center justify-center py-2 lg:py-0">
            <div className="bg-indigo-600 text-white rounded-full px-3 py-2 text-xs font-bold shadow-lg flex items-center gap-1">
              <GitBranch className="w-3 h-3" /> FUSION
            </div>
          </div>
          <div className="flex-1 bg-emerald-50 rounded-xl p-3 border border-emerald-100 text-center">
            <Layers className="w-7 h-7 text-emerald-600 mx-auto mb-1" />
            <p className="font-bold text-emerald-800 text-xs">Audio CNN Branch</p>
            <p className="text-xs text-emerald-500">128×128 Mel-Spectrogram</p>
            <p className="text-xs text-emerald-400">Dementia Audio (Kaggle)</p>
          </div>
          <div className="flex-1 bg-green-50 rounded-xl p-3 border border-green-100 text-center">
            <Layers className="w-7 h-7 text-green-600 mx-auto mb-1" />
            <p className="font-bold text-green-800 text-xs">ANN Branch</p>
            <p className="text-xs text-green-500">10 Clinical Features</p>
            <p className="text-xs text-green-400">NHANES / GDS / Body Fat</p>
          </div>
        </div>

        <div className="text-center bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-xl py-3 px-4 text-sm font-semibold">
          Concatenated Multi-Modal Vector → Dense(128) → Dense(64) → Softmax(5) → 5 Syndrome Probabilities
        </div>
      </div>

      {/* Detailed layer breakdowns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <BranchSection
          branch={arch.cnn_branch}
          color="#3b82f6"
          icon={<div className="p-2 bg-blue-50 rounded-xl"><Cpu className="w-5 h-5 text-blue-600" /></div>}
        />
        <BranchSection
          branch={arch.ann_branch}
          color="#10b981"
          icon={<div className="p-2 bg-green-50 rounded-xl"><Layers className="w-5 h-5 text-green-600" /></div>}
        />
      </div>

      {/* Fusion section */}
      <BranchSection
        branch={arch.fusion}
        color="#6366f1"
        icon={<div className="p-2 bg-indigo-50 rounded-xl"><GitBranch className="w-5 h-5 text-indigo-600" /></div>}
      />

      {/* Architecture explanation */}
      <div className="bg-indigo-50 border border-indigo-100 rounded-2xl p-6">
        <h3 className="font-bold text-indigo-800 mb-3">Why Hybrid Fusion Improves Performance</h3>
        <div className="text-sm text-indigo-700 space-y-3 leading-relaxed">
          <p>
            Geriatric syndrome assessment inherently involves both visual cues (posture abnormalities, facial affect,
            gait deviations) and quantitative clinical measurements (blood pressure, cognitive test scores, mobility indices).
            No single modality captures the full diagnostic picture.
          </p>
          <p>
            A patient may have normal vital signs but exhibit significant postural instability visible only in imagery.
            Conversely, a patient may appear physically normal while their cognitive scores reveal early impairment.
            By fusing CNN and ANN outputs at a learned intermediate layer, the model discovers latent correlations
            between visual and clinical signals that would be invisible to either branch in isolation.
          </p>
          <p>
            The concatenation-based fusion strategy was chosen over alternatives (attention-based, bilinear pooling)
            for its interpretability and training stability on the limited dataset size. The subsequent dense layers
            (128 → 64 → 5) learn non-linear cross-modal interactions before producing the final classification.
          </p>
        </div>
      </div>
    </div>
  )
}
