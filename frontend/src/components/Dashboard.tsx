import React, { useState, useEffect } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell
} from 'recharts'
import { PhenotypingReport, ModelMetrics, ModelType, MODEL_DISPLAY_NAMES } from '../types'
import { getMetrics, MOCK_METRICS } from '../services/apiService'
import {
  Activity, Brain, ChevronRight, CheckCircle, AlertTriangle, TrendingUp,
  BarChart2, Target, Layers, Award, RotateCcw, Image as ImageIcon, Eye
} from 'lucide-react'

const CLASS_COLORS = ['#ef4444', '#f59e0b', '#6366f1', '#10b981', '#8b5cf6']

interface Props { report: PhenotypingReport; onReset: () => void }

function RiskBadge({ score }: { score: number }) {
  const cfg = score > 70 ? { bg: 'bg-red-50 border-red-200 text-red-700', label: 'High' }
    : score > 40 ? { bg: 'bg-amber-50 border-amber-200 text-amber-700', label: 'Mod' }
    : { bg: 'bg-green-50 border-green-200 text-green-700', label: 'Low' }
  return <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${cfg.bg}`}>{cfg.label} {score}%</span>
}

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color: string }) {
  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-5">
      <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-3xl font-black ${color}`}>{value}</p>
      {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
    </div>
  )
}

function SectionTab({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick}
      className={`px-4 py-2 rounded-xl text-sm font-semibold transition-all ${active ? 'bg-indigo-600 text-white shadow' : 'bg-slate-100 text-slate-500 hover:bg-slate-200'}`}>
      {children}
    </button>
  )
}

function ConfusionMatrix({ metrics }: { metrics: ModelMetrics }) {
  const max = Math.max(...metrics.confusionMatrix.flat())
  const labels = metrics.perClass.map(p => p.name.split(' ')[0])
  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
      <div className="flex items-center gap-2 mb-5">
        <Layers className="w-5 h-5 text-indigo-600" />
        <div>
          <h3 className="font-bold text-slate-800">Confusion Matrix</h3>
          <p className="text-xs text-slate-400">{metrics.modelName} · 5-class</p>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-center text-sm">
          <thead>
            <tr>
              <th className="p-1 text-slate-400 text-xs">Act ↓ / Pred →</th>
              {labels.map((l, i) => <th key={i} className="p-1 text-xs font-bold" style={{ color: CLASS_COLORS[i] }}>{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {metrics.confusionMatrix.map((row, ri) => (
              <tr key={ri}>
                <td className="p-1 text-xs font-bold text-right pr-2" style={{ color: CLASS_COLORS[ri] }}>{labels[ri]}</td>
                {row.map((val, ci) => {
                  const isDiag = ri === ci
                  const bg = isDiag
                    ? `rgba(99,102,241,${0.15 + (val / max) * 0.75})`
                    : val > 4 ? `rgba(239,68,68,${0.1 + (val / max) * 0.4})` : 'rgba(241,245,249,0.6)'
                  return (
                    <td key={ci} className="p-2.5 rounded-lg font-bold text-sm"
                      style={{ background: bg, color: isDiag ? '#3730a3' : val > 4 ? '#991b1b' : '#64748b' }}>
                      {val}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function MetricsTable({ metrics }: { metrics: ModelMetrics }) {
  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
      <div className="flex items-center gap-2 mb-5">
        <Target className="w-5 h-5 text-indigo-600" />
        <div>
          <h3 className="font-bold text-slate-800">Per-Class Metrics</h3>
          <p className="text-xs text-slate-400">Precision · Recall · Specificity · F1-Score</p>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100">
              <th className="text-left py-2 text-xs font-bold text-slate-400 uppercase">Class</th>
              {['Precision', 'Recall', 'Specificity', 'F1'].map(h => (
                <th key={h} className="text-right py-2 text-xs font-bold text-slate-400 uppercase px-2">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.perClass.map((m, i) => (
              <tr key={i} className="border-b border-slate-50 hover:bg-slate-50">
                <td className="py-3 font-semibold text-sm" style={{ color: CLASS_COLORS[i] }}>{m.name}</td>
                {[m.precision, m.recall, m.specificity, m.f1].map((v, vi) => (
                  <td key={vi} className="py-3 text-right px-2">
                    <span className={`font-bold text-sm ${v >= 95 ? 'text-green-600' : v >= 90 ? 'text-indigo-600' : 'text-amber-500'}`}>{v}%</span>
                    <div className="w-full bg-slate-100 rounded-full h-1 mt-1"><div className="h-1 rounded-full bg-indigo-400" style={{ width: `${v}%`, opacity: 0.6 }} /></div>
                  </td>
                ))}
              </tr>
            ))}
            <tr className="bg-indigo-50 font-bold">
              <td className="py-2.5 text-indigo-700 text-xs uppercase tracking-wide">Macro Avg</td>
              {[metrics.precision, metrics.recall,
                +(metrics.perClass.reduce((s, m) => s + m.specificity, 0) / 5).toFixed(1),
                metrics.f1Score].map((v, vi) => (
                <td key={vi} className="py-2.5 text-right px-2 text-indigo-700 font-black">{v}%</td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

function TrainingCurves({ metrics }: { metrics: ModelMetrics }) {
  const [mode, setMode] = useState<'loss' | 'acc'>('loss')
  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-indigo-600" />
          <div>
            <h3 className="font-bold text-slate-800">Training & Validation Curves</h3>
            <p className="text-xs text-slate-400">{metrics.modelName} · 30 epochs</p>
          </div>
        </div>
        <div className="flex gap-2">
          {(['loss', 'acc'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${mode === m ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-500'}`}>
              {m === 'loss' ? 'Loss' : 'Accuracy'}
            </button>
          ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={metrics.trainingCurves} margin={{ top: 5, right: 20, left: 0, bottom: 32 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey="epoch"
            ticks={[1, 5, 10, 15, 20, 25, 30]}
            tick={{ fontSize: 10, fill: '#94a3b8' }}
            label={{ value: 'Epoch', position: 'insideBottom', offset: -12, fontSize: 11, fill: '#64748b', fontWeight: 600 }}
          />
          <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} domain={mode === 'loss' ? [0, 0.6] : [20, 100]} />
          <Tooltip contentStyle={{ borderRadius: '10px', border: '1px solid #e2e8f0', fontSize: 11 }} />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 18 }} />
          {mode === 'loss' ? (<>
            <Line type="monotone" dataKey="trainLoss" name="Train Loss" stroke="#6366f1" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="valLoss" name="Val Loss" stroke="#f59e0b" strokeWidth={2} dot={false} strokeDasharray="5 3" />
          </>) : (<>
            <Line type="monotone" dataKey="trainAcc" name="Train Acc %" stroke="#10b981" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="valAcc" name="Val Acc %" stroke="#8b5cf6" strokeWidth={2} dot={false} strokeDasharray="5 3" />
          </>)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function Dashboard({ report, onReset }: Props) {
  const [tab, setTab] = useState<'clinical' | 'performance'>('clinical')

  // Determine which model was used to select correct metrics
  const modelKey: ModelType = report.modelUsed.toLowerCase().includes('resnet') ? 'resnet50'
    : report.modelUsed.toLowerCase().includes('mobile') ? 'mobilenetv2' : 'custom'
  const metrics = MOCK_METRICS[modelKey]

  const radarData = report.assessments.map(a => ({
    subject: a.syndrome.split(' ')[0],
    score: a.riskScore,
    fullMark: 100
  }))

  const barData = report.assessments.map((a, i) => ({
    name: a.syndrome.split(' ')[0],
    risk: a.riskScore,
    conf: a.confidence,
    color: CLASS_COLORS[i]
  }))

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6 fade-up">
      {/* Top bar */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <span className="text-sm font-bold text-green-600">Analysis Complete</span>
            <span className="inline-flex items-center gap-1 text-xs bg-indigo-50 text-indigo-600 border border-indigo-200 px-2 py-0.5 rounded-full font-semibold">
              {report.modelUsed}
            </span>
            {report.hasImage && (
              <span className="inline-flex items-center gap-1 text-xs bg-blue-50 text-blue-600 border border-blue-200 px-2 py-0.5 rounded-full font-semibold">
                <ImageIcon className="w-3 h-3" /> Image
              </span>
            )}
          </div>
          <h1 className="text-2xl font-extrabold text-slate-900">Hybrid CNN-ANN Phenotyping Report</h1>
        </div>
        <button onClick={onReset}
          className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 text-slate-600 rounded-xl hover:bg-slate-50 text-sm font-semibold shadow-sm transition-all">
          <RotateCcw className="w-4 h-4" /> New Analysis
        </button>
      </div>

      {/* Tab switch */}
      <div className="flex gap-2 p-1 bg-slate-100 rounded-2xl w-fit">
        <SectionTab active={tab === 'clinical'} onClick={() => setTab('clinical')}>🩺 Clinical Report</SectionTab>
        <SectionTab active={tab === 'performance'} onClick={() => setTab('performance')}>📊 Model Performance</SectionTab>
      </div>

      {/* ─── CLINICAL TAB ─── */}
      {tab === 'clinical' && (
        <div className="space-y-6">
          {/* Risk score cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {report.assessments.map((a, i) => (
              <div key={i} className="bg-white rounded-2xl border border-slate-100 shadow-sm p-4 text-center">
                <p className="text-xs font-bold text-slate-400 mb-1">{a.syndrome.split(' ')[0]}</p>
                <p className="text-3xl font-black" style={{ color: CLASS_COLORS[i] }}>{a.riskScore}%</p>
                <div className="w-full bg-slate-100 rounded-full h-1.5 mt-2">
                  <div className="h-1.5 rounded-full transition-all duration-1000" style={{ width: `${a.riskScore}%`, background: CLASS_COLORS[i] }} />
                </div>
                <RiskBadge score={a.riskScore} />
              </div>
            ))}
          </div>

          {/* Summary + Radar */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-2 bg-indigo-50 rounded-xl"><Activity className="w-5 h-5 text-indigo-600" /></div>
                <h2 className="text-lg font-bold">Hybrid CNN-ANN Clinical Summary</h2>
              </div>
              <p className="text-slate-600 leading-relaxed mb-6 text-sm">{report.summary}</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-slate-50 rounded-xl p-4">
                  <p className="text-xs font-bold text-slate-400 uppercase mb-2">Visual Phenotype (CNN Branch)</p>
                  <p className="text-xs text-indigo-600 font-semibold mb-1.5">{report.visualPhenotype.postureAnalysis}</p>
                  <ul className="space-y-1">
                    {report.visualPhenotype.gaitMarkers.map((m, i) => (
                      <li key={i} className="flex gap-2 text-xs text-slate-600"><ChevronRight className="w-3 h-3 text-indigo-400 mt-0.5 flex-shrink-0" />{m}</li>
                    ))}
                  </ul>
                </div>
                <div className="bg-slate-50 rounded-xl p-4">
                  <p className="text-xs font-bold text-slate-400 uppercase mb-2">Clinical Markers (ANN Branch)</p>
                  <p className="text-xs font-semibold text-indigo-600 mb-1">Cognitive Assessment:</p>
                  <p className="text-xs text-slate-600 mb-3">{report.neuralMarkers.speechCoherence}</p>
                  <p className="text-xs font-semibold text-indigo-600 mb-1">Clinical Integration:</p>
                  <p className="text-xs text-slate-600">{report.neuralMarkers.cognitiveLoad}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 flex flex-col items-center">
              <h2 className="text-lg font-bold self-start mb-1">Risk Profile</h2>
              <p className="text-xs text-slate-400 self-start mb-4">Syndrome risk radar</p>
              <div className="w-full h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
                    <PolarGrid stroke="#e2e8f0" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 10 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar name="Risk" dataKey="score" stroke="#6366f1" fill="#6366f1" fillOpacity={0.55} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Grad-CAM */}
          {report.gradCamBase64 && (
            <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
              <div className="flex items-center gap-2 mb-4">
                <Eye className="w-5 h-5 text-indigo-600" />
                <div>
                  <h3 className="font-bold text-slate-800">Grad-CAM Explainability</h3>
                  <p className="text-xs text-slate-400">Regions that most strongly influenced the CNN classification</p>
                </div>
              </div>
              <div className="flex justify-center">
                <img src={`data:image/png;base64,${report.gradCamBase64}`} alt="Grad-CAM"
                  className="rounded-xl border border-slate-200 max-w-xs shadow-lg" />
              </div>
              <p className="text-xs text-slate-500 text-center mt-3">
                Warmer colours indicate regions with higher activation importance for the predicted syndrome class.
              </p>
            </div>
          )}

          {/* Risk bar chart */}
          <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
            <h2 className="font-bold text-slate-800 mb-1">Risk Score vs Confidence</h2>
            <p className="text-xs text-slate-400 mb-4">Per-syndrome comparison</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={barData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#94a3b8' }} />
                <Tooltip contentStyle={{ borderRadius: '10px', fontSize: 12 }} formatter={(v: number) => `${v}%`} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="risk" name="Risk Score %" radius={[6, 6, 0, 0]}>
                  {barData.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.85} />)}
                </Bar>
                <Bar dataKey="conf" name="Confidence %" fill="#cbd5e1" radius={[6, 6, 0, 0]} fillOpacity={0.6} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Detailed assessment cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {report.assessments.map((item, idx) => (
              <div key={idx} className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden flex flex-col">
                <div className="p-4 border-b" style={{ background: `${CLASS_COLORS[idx]}15`, borderColor: `${CLASS_COLORS[idx]}30` }}>
                  <div className="flex justify-between items-center mb-2">
                    <h3 className="font-bold text-slate-800">{item.syndrome}</h3>
                    <span className="text-xl font-black" style={{ color: CLASS_COLORS[idx] }}>{item.riskScore}%</span>
                  </div>
                  <div className="w-full bg-white/60 rounded-full h-2 overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-1000" style={{ width: `${item.riskScore}%`, background: CLASS_COLORS[idx] }} />
                  </div>
                  <div className="flex justify-between mt-1.5 text-xs text-slate-500">
                    <span>Confidence: {item.confidence}%</span>
                    <RiskBadge score={item.riskScore} />
                  </div>
                </div>
                <div className="p-4 flex-grow space-y-3">
                  <div>
                    <p className="text-xs font-bold text-slate-400 uppercase mb-1.5">Observations</p>
                    <ul className="space-y-1">
                      {item.observations.map((obs, i) => (
                        <li key={i} className="text-xs text-slate-600 flex gap-1.5"><span style={{ color: CLASS_COLORS[idx] }}>•</span>{obs}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <p className="text-xs font-bold text-slate-400 uppercase mb-1.5">Recommendations</p>
                    <ul className="space-y-1">
                      {item.recommendations.map((rec, i) => (
                        <li key={i} className="text-xs text-slate-600 flex gap-1.5"><CheckCircle className="w-3 h-3 flex-shrink-0 mt-0.5" style={{ color: CLASS_COLORS[idx] }} />{rec}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* CTA to performance */}
          <div className="bg-indigo-50 border border-indigo-100 rounded-2xl p-6 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Brain className="w-8 h-8 text-indigo-600" />
              <div>
                <p className="font-bold text-indigo-800">{metrics.modelName} Accuracy: {metrics.accuracy}%</p>
                <p className="text-xs text-indigo-600">View confusion matrix, per-class metrics, training curves</p>
              </div>
            </div>
            <button onClick={() => setTab('performance')}
              className="px-5 py-2.5 bg-indigo-600 text-white rounded-xl font-semibold text-sm hover:bg-indigo-700 transition-all whitespace-nowrap shadow">
              View Performance →
            </button>
          </div>
        </div>
      )}

      {/* ─── PERFORMANCE TAB ─── */}
      {tab === 'performance' && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Overall Accuracy" value={`${metrics.accuracy}%`} sub={`${metrics.modelName}`} color="text-indigo-600" />
            <StatCard label="Macro F1-Score" value={`${metrics.f1Score}%`} sub="Avg across 5 classes" color="text-green-600" />
            <StatCard label="Macro Precision" value={`${metrics.precision}%`} sub="Avg across 5 classes" color="text-blue-600" />
            <StatCard label="Macro Recall" value={`${metrics.recall}%`} sub="Sensitivity (avg)" color="text-purple-600" />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <ConfusionMatrix metrics={metrics} />
            <MetricsTable metrics={metrics} />
          </div>

          <TrainingCurves metrics={metrics} />

          {/* Architecture config */}
          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6">
            <p className="font-bold text-slate-700 mb-3 text-sm">Model Architecture & Training Config</p>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 text-xs">
              {[
                { k: 'Architecture', v: metrics.modelName },
                { k: 'Input Size', v: '224 × 224 × 3 RGB' },
                { k: 'Fusion', v: 'CNN + ANN Concatenation' },
                { k: 'Output Head', v: '5-class Softmax' },
                { k: 'Optimizer', v: 'Adam lr=0.0001' },
                { k: 'Batch / Epochs', v: '32 / 30' },
                { k: 'Loss', v: 'Categorical Cross-Entropy' },
                { k: 'Regularization', v: 'Dropout + L2 + BatchNorm' },
              ].map(({ k, v }) => (
                <div key={k} className="bg-white rounded-xl border border-slate-100 p-3">
                  <p className="font-semibold text-slate-400">{k}</p>
                  <p className="font-black text-slate-800 mt-0.5">{v}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
