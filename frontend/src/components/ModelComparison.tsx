import React, { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, LineChart, Line, Cell
} from 'recharts'
import { BarChart2, TrendingUp, Layers, Award, Zap, HardDrive, Clock, Cpu } from 'lucide-react'
import { ModelType, MODEL_DISPLAY_NAMES, ModelMetrics } from '../types'
import { getAllMetrics } from '../services/apiService'

const MODEL_COLORS: Record<ModelType, string> = {
  resnet50: '#6366f1',
  mobilenetv2: '#8b5cf6',
  efficientnet: '#10b981',
  custom: '#ec4899',
}

const SYNDROMES = ['Frailty', 'Sarcopenia', 'Cognitive Imp.', 'Fall Risk', 'Depression']

export default function ModelComparison() {
  const [metrics, setMetrics] = useState<Record<ModelType, ModelMetrics> | null>(null)
  const [selectedCurve, setSelectedCurve] = useState<ModelType>('resnet50')
  const [curveMode, setCurveMode] = useState<'loss' | 'acc'>('loss')

  useEffect(() => { getAllMetrics().then(setMetrics) }, [])

  if (!metrics) return <div className="py-20 text-center text-slate-400">Loading comparison data...</div>

  const models: ModelType[] = ['resnet50', 'mobilenetv2', 'custom']

  // Comparison bar data
  const comparisonData = [
    { metric: 'Accuracy', ...Object.fromEntries(models.map(m => [m, metrics[m].accuracy])) },
    { metric: 'Precision', ...Object.fromEntries(models.map(m => [m, metrics[m].precision])) },
    { metric: 'Recall', ...Object.fromEntries(models.map(m => [m, metrics[m].recall])) },
    { metric: 'F1-Score', ...Object.fromEntries(models.map(m => [m, metrics[m].f1Score])) },
  ]

  // Radar data per class (for selected model pair)
  const radarData = SYNDROMES.map((s, i) => ({
    syndrome: s,
    ...Object.fromEntries(models.map(m => [m, metrics[m].perClass[i]?.f1 ?? 0]))
  }))

  return (
    <div className="max-w-6xl mx-auto py-12 px-4 space-y-8 fade-up">
      {/* Header */}
      <div className="text-center">
        <span className="inline-block bg-indigo-50 text-indigo-700 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider mb-3">Performance Analysis</span>
        <h1 className="text-4xl font-extrabold text-slate-900 mb-3">Model Comparison</h1>
        <p className="text-slate-500 max-w-xl mx-auto">Side-by-side evaluation of all three hybrid CNN-ANN architectures across key performance metrics.</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {models.map(m => {
          const d = metrics[m]
          return (
            <div key={m} className="bg-white rounded-2xl border-2 shadow-sm p-6 hover:shadow-lg transition-all" style={{ borderColor: `${MODEL_COLORS[m]}30` }}>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-bold px-2.5 py-1 rounded-full" style={{ background: `${MODEL_COLORS[m]}15`, color: MODEL_COLORS[m] }}>
                  {MODEL_DISPLAY_NAMES[m]}
                </span>
                <span className="text-3xl font-black" style={{ color: MODEL_COLORS[m] }}>{d.accuracy}%</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs mb-4">
                <div className="bg-slate-50 rounded-lg p-2"><p className="text-slate-400">Precision</p><p className="font-bold text-slate-700">{d.precision}%</p></div>
                <div className="bg-slate-50 rounded-lg p-2"><p className="text-slate-400">Recall</p><p className="font-bold text-slate-700">{d.recall}%</p></div>
                <div className="bg-slate-50 rounded-lg p-2"><p className="text-slate-400">F1-Score</p><p className="font-bold text-slate-700">{d.f1Score}%</p></div>
                <div className="bg-slate-50 rounded-lg p-2"><p className="text-slate-400">Mean AUC</p><p className="font-bold text-slate-700">{(d.rocAuc.reduce((a, b) => a + b, 0) / 5).toFixed(3)}</p></div>
              </div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="flex items-center gap-1 text-slate-500"><HardDrive className="w-3 h-3" /> {d.modelSizeMb} MB</div>
                <div className="flex items-center gap-1 text-slate-500"><Clock className="w-3 h-3" /> {d.inferenceTimeMs} ms</div>
                <div className="flex items-center gap-1 text-slate-500"><Cpu className="w-3 h-3" /> {(d.totalParams / 1e6).toFixed(1)}M</div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Full comparison table */}
      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
        <div className="flex items-center gap-2 mb-5">
          <BarChart2 className="w-5 h-5 text-indigo-600" />
          <h2 className="font-bold text-slate-800">Comprehensive Performance Table</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left py-3 text-xs font-bold text-slate-400 uppercase">Metric</th>
                {models.map(m => (
                  <th key={m} className="text-center py-3 text-xs font-bold uppercase" style={{ color: MODEL_COLORS[m] }}>{MODEL_DISPLAY_NAMES[m]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { label: 'Accuracy (%)', key: 'accuracy' },
                { label: 'Precision (%)', key: 'precision' },
                { label: 'Recall (%)', key: 'recall' },
                { label: 'F1-Score (%)', key: 'f1Score' },
                { label: 'Training Time (min)', key: 'trainingTimeMin' },
                { label: 'Model Size (MB)', key: 'modelSizeMb' },
                { label: 'Inference (ms)', key: 'inferenceTimeMs' },
                { label: 'Parameters', key: 'totalParams' },
              ].map(row => (
                <tr key={row.label} className="border-b border-slate-50 hover:bg-slate-50">
                  <td className="py-3 font-semibold text-slate-700">{row.label}</td>
                  {models.map(m => {
                    const val = (metrics[m] as any)[row.key]
                    const display = row.key === 'totalParams' ? `${(val / 1e6).toFixed(1)}M` : val
                    const isBest = models.every(o => {
                      const ov = (metrics[o] as any)[row.key]
                      if (['modelSizeMb', 'inferenceTimeMs', 'trainingTimeMin', 'totalParams'].includes(row.key)) return val <= ov
                      return val >= ov
                    })
                    return (
                      <td key={m} className="py-3 text-center">
                        <span className={`font-bold ${isBest ? 'text-emerald-600' : 'text-slate-600'}`}>
                          {display}{isBest && ' ✓'}
                        </span>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Comparison bar chart + Radar */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
          <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2"><BarChart2 className="w-5 h-5 text-indigo-600" /> Metric Comparison</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={comparisonData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="metric" tick={{ fontSize: 11, fill: '#94a3b8' }} />
              <YAxis domain={[85, 100]} tick={{ fontSize: 11, fill: '#94a3b8' }} />
              <Tooltip contentStyle={{ borderRadius: '10px', fontSize: 12 }} formatter={(v: number) => `${v}%`} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {models.map(m => (
                <Bar key={m} dataKey={m} name={MODEL_DISPLAY_NAMES[m]} fill={MODEL_COLORS[m]} radius={[4, 4, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
          <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2"><Award className="w-5 h-5 text-indigo-600" /> Per-Class F1 Radar</h3>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="syndrome" tick={{ fill: '#64748b', fontSize: 10 }} />
              <PolarRadiusAxis domain={[85, 100]} tick={false} axisLine={false} />
              {models.map(m => (
                <Radar key={m} name={MODEL_DISPLAY_NAMES[m]} dataKey={m} stroke={MODEL_COLORS[m]} fill={MODEL_COLORS[m]} fillOpacity={0.15} strokeWidth={2} />
              ))}
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Training curves comparison */}
      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-indigo-600" />
            <div>
              <h3 className="font-bold text-slate-800">Training Curves</h3>
              <p className="text-xs text-slate-400">30 epochs · {MODEL_DISPLAY_NAMES[selectedCurve]}</p>
            </div>
          </div>
          <div className="flex gap-2">
            {models.map(m => (
              <button key={m} onClick={() => setSelectedCurve(m)}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${selectedCurve === m ? 'text-white shadow' : 'bg-slate-100 text-slate-500'}`}
                style={selectedCurve === m ? { background: MODEL_COLORS[m] } : {}}>
                {MODEL_DISPLAY_NAMES[m].split(' ')[0]}
              </button>
            ))}
            <div className="w-px bg-slate-200 mx-1" />
            {(['loss', 'acc'] as const).map(mode => (
              <button key={mode} onClick={() => setCurveMode(mode)}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${curveMode === mode ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-500'}`}>
                {mode === 'loss' ? 'Loss' : 'Accuracy'}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={290}>
          <LineChart data={metrics[selectedCurve].trainingCurves} margin={{ top: 5, right: 20, left: 0, bottom: 32 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              dataKey="epoch"
              ticks={[1, 5, 10, 15, 20, 25, 30]}
              tick={{ fontSize: 10, fill: '#94a3b8' }}
              label={{ value: 'Epoch', position: 'insideBottom', offset: -12, fontSize: 11, fill: '#64748b', fontWeight: 600 }}
            />
            <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} domain={curveMode === 'loss' ? [0, 0.6] : [20, 100]} />
            <Tooltip contentStyle={{ borderRadius: '10px', border: '1px solid #e2e8f0', fontSize: 11 }} />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 18 }} />
            {curveMode === 'loss' ? (<>
              <Line type="monotone" dataKey="trainLoss" name="Train Loss" stroke={MODEL_COLORS[selectedCurve]} strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="valLoss" name="Val Loss" stroke="#f59e0b" strokeWidth={2} dot={false} strokeDasharray="5 3" />
            </>) : (<>
              <Line type="monotone" dataKey="trainAcc" name="Train Acc %" stroke={MODEL_COLORS[selectedCurve]} strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="valAcc" name="Val Acc %" stroke="#10b981" strokeWidth={2} dot={false} strokeDasharray="5 3" />
            </>)}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Analysis commentary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-indigo-50 border border-indigo-100 rounded-2xl p-5">
          <h4 className="font-bold text-indigo-800 mb-2 flex items-center gap-2"><Zap className="w-4 h-4" /> ResNet50 — High Accuracy</h4>
          <p className="text-sm text-indigo-700 leading-relaxed">
            Achieves the highest accuracy (95.8%) thanks to its 50-layer deep architecture with skip connections 
            that capture fine-grained visual hierarchies. ImageNet pre-training provides a strong feature 
            initialisation. The trade-off is a larger model footprint (98.2 MB) and slower inference (45.3 ms), 
            making it best suited for server-side deployment where computational resources are available.
          </p>
        </div>
        <div className="bg-purple-50 border border-purple-100 rounded-2xl p-5">
          <h4 className="font-bold text-purple-800 mb-2 flex items-center gap-2"><Zap className="w-4 h-4" /> MobileNetV2 — Efficient</h4>
          <p className="text-sm text-purple-700 leading-relaxed">
            Depth-wise separable convolutions reduce model size to 14.1 MB (~7× smaller) while retaining 93.2% 
            accuracy.  Inference completes in 12.8 ms — fast enough for real-time mobile applications.  
            The inverted residual blocks with linear bottlenecks maintain representational efficiency.  
            Ideal for edge deployment scenarios such as clinic tablets or low-power devices.
          </p>
        </div>
        <div className="bg-pink-50 border border-pink-100 rounded-2xl p-5">
          <h4 className="font-bold text-pink-800 mb-2 flex items-center gap-2"><Zap className="w-4 h-4" /> GeriatricNet — Balanced</h4>
          <p className="text-sm text-pink-700 leading-relaxed">
            The custom-designed architecture achieves 91.5% accuracy with only 2.1M parameters and 8.6 MB 
            storage.  The progressive filter expansion (32→64→128→256) was deliberately tuned to balance 
            capacity against the limited dataset size.  Despite being trained from scratch without pre-trained 
            weights, it demonstrates competitive per-class F1 scores, validating the domain-specific design 
            choices.
          </p>
        </div>
      </div>

      {/* Confusion matrices */}
      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
        <div className="flex items-center gap-2 mb-5">
          <Layers className="w-5 h-5 text-indigo-600" />
          <h2 className="font-bold text-slate-800">Confusion Matrices — All Models</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {models.map(m => {
            const cm = metrics[m].confusionMatrix
            const max = Math.max(...cm.flat())
            return (
              <div key={m}>
                <p className="text-sm font-bold mb-2" style={{ color: MODEL_COLORS[m] }}>{MODEL_DISPLAY_NAMES[m]}</p>
                <table className="w-full text-center text-xs">
                  <thead>
                    <tr>
                      <th className="p-1 text-slate-400">↓Actual/Pred→</th>
                      {['Fr', 'Sa', 'Co', 'Fa', 'De'].map((l, i) => <th key={i} className="p-1 font-bold text-slate-500">{l}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {cm.map((row, ri) => (
                      <tr key={ri}>
                        <td className="p-1 text-xs font-bold text-slate-500 text-right">{['Fr', 'Sa', 'Co', 'Fa', 'De'][ri]}</td>
                        {row.map((val, ci) => {
                          const isDiag = ri === ci
                          const bg = isDiag
                            ? `rgba(99,102,241,${0.15 + (val / max) * 0.75})`
                            : val > 4 ? `rgba(239,68,68,${0.1 + (val / max) * 0.4})` : 'rgba(241,245,249,0.6)'
                          return (
                            <td key={ci} className="p-1.5 rounded font-bold"
                              style={{ background: bg, color: isDiag ? '#3730a3' : val > 4 ? '#991b1b' : '#94a3b8' }}>
                              {val}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
