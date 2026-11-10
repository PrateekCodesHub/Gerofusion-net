import React from 'react'
import { Brain, Layers, BarChart2, Cpu, ArrowRight, Zap, Shield, GitBranch } from 'lucide-react'
import { PageView } from '../types'

interface Props { onNavigate: (page: PageView) => void }

export default function LandingHero({ onNavigate }: Props) {
  return (
    <div className="min-h-[calc(100vh-64px)]">

      {/* Hero */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900 py-24 px-4">
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-20 left-20 w-72 h-72 bg-indigo-500 rounded-full blur-[120px]" />
          <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500 rounded-full blur-[150px]" />
        </div>
        <div className="max-w-5xl mx-auto relative z-10 text-center">
          <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-md px-4 py-2 rounded-full mb-6 border border-white/10">
            <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
            <span className="text-sm text-indigo-200 font-medium">Hybrid CNN-ANN Architecture · Multi-Modal Fusion</span>
          </div>
          <h1 className="text-5xl md:text-7xl font-black text-white mb-6 leading-tight tracking-tight">
            Geriatric Syndrome<br />
            <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">GeroFusion</span>
          </h1>
          <p className="text-lg md:text-xl text-indigo-200/80 max-w-2xl mx-auto mb-10 leading-relaxed">
            An advanced deep learning system for multi-syndrome geriatric assessment. 
            Fuses convolutional visual features with structured clinical data through a 
            novel hybrid architecture to deliver explainable, real-time diagnostic insights.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <button onClick={() => onNavigate('analysis')}
              className="px-8 py-4 bg-indigo-600 text-white rounded-2xl font-bold text-lg hover:bg-indigo-500 active:scale-[0.98] transition-all flex items-center gap-2 shadow-xl shadow-indigo-900/50">
              <Brain className="w-5 h-5" /> Begin Analysis <ArrowRight className="w-5 h-5" />
            </button>
            <button onClick={() => onNavigate('architecture')}
              className="px-8 py-4 bg-white/10 text-white rounded-2xl font-bold text-lg hover:bg-white/20 border border-white/20 transition-all flex items-center gap-2 backdrop-blur-md">
              <Layers className="w-5 h-5" /> Explore Architecture
            </button>
          </div>
        </div>
      </section>

      {/* Stats bar */}
      <section className="bg-slate-900 border-y border-white/10 py-6 px-4">
        <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
          {[
            { label: 'Peak Accuracy', value: '95.8%', sub: 'ResNet50 Hybrid' },
            { label: 'Model Variants', value: '3', sub: 'CNN Backbones' },
            { label: 'Clinical Features', value: '10', sub: 'Tabular Inputs' },
            { label: 'Syndromes', value: '5', sub: 'Classification' },
          ].map(s => (
            <div key={s.label}>
              <p className="text-3xl font-black text-white">{s.value}</p>
              <p className="text-xs text-indigo-300 font-semibold mt-1">{s.label}</p>
              <p className="text-xs text-slate-500">{s.sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Feature cards */}
      <section className="py-20 px-4 bg-slate-50">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <span className="inline-block bg-indigo-50 text-indigo-700 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider mb-3">System Capabilities</span>
            <h2 className="text-3xl font-extrabold text-slate-900 mb-3">Multi-Modal Deep Learning Pipeline</h2>
            <p className="text-slate-500 max-w-xl mx-auto">A full-stack system combining computer vision, tabular data processing, and feature fusion for comprehensive geriatric assessment.</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              { icon: Layers, title: 'Hybrid CNN-ANN Fusion', desc: 'Concatenation-based feature fusion merges visual CNN embeddings with clinical ANN encodings for multi-modal classification.', color: 'indigo' },
              { icon: Cpu, title: '3 CNN Backbones', desc: 'ResNet50 (transfer learning), MobileNetV2 (edge deployment), and a custom GeriatricNet designed for this domain.', color: 'purple' },
              { icon: BarChart2, title: 'Comprehensive Metrics', desc: 'Per-class precision, recall, F1, confusion matrix, ROC-AUC, training curves, and cross-model comparison.', color: 'blue' },
              { icon: Zap, title: 'Real-Time Inference', desc: 'FastAPI backend serves predictions in <50ms. REST API architecture supports concurrent requests.', color: 'amber' },
              { icon: Shield, title: 'Grad-CAM Explainability', desc: 'Gradient-weighted Class Activation Maps highlight which image regions drive the model predicted diagnosis.', color: 'emerald' },
              { icon: GitBranch, title: 'Full-Stack Architecture', desc: 'React frontend + FastAPI backend + optional database. Clear separation of concerns with REST API communication.', color: 'rose' },
            ].map(card => (
              <div key={card.title} className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 hover:shadow-lg hover:-translate-y-1 transition-all duration-300">
                <div className={`inline-flex p-3 bg-${card.color}-50 rounded-xl mb-4`}>
                  <card.icon className={`w-6 h-6 text-${card.color}-600`} />
                </div>
                <h3 className="text-lg font-bold text-slate-800 mb-2">{card.title}</h3>
                <p className="text-sm text-slate-500 leading-relaxed">{card.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Model comparison teaser */}
      <section className="py-20 px-4 bg-white">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <span className="inline-block bg-purple-50 text-purple-700 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider mb-3">Model Comparison</span>
            <h2 className="text-3xl font-extrabold text-slate-900 mb-3">Three Architectures, One Goal</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { name: 'ResNet50 Hybrid', acc: 95.8, size: '98.2 MB', speed: '45.3 ms', params: '24.9M', tag: 'Highest Accuracy', color: '#6366f1', desc: 'Deep residual network with skip connections. ImageNet pre-trained. Top 20 layers fine-tuned.' },
              { name: 'MobileNetV2 Hybrid', acc: 93.2, size: '14.1 MB', speed: '12.8 ms', params: '3.5M', tag: 'Edge Optimised', color: '#8b5cf6', desc: 'Depth-wise separable convolutions. 7× smaller than ResNet50. Ideal for mobile deployment.' },
              { name: 'GeriatricNet (Custom)', acc: 91.5, size: '8.6 MB', speed: '8.4 ms', params: '2.1M', tag: 'Original Design', color: '#ec4899', desc: 'Custom 4-block CNN designed for this project. Progressive filter expansion (32→64→128→256).' },
            ].map((m, i) => (
              <div key={m.name} className="bg-white rounded-2xl border-2 shadow-sm p-6 hover:shadow-xl transition-all duration-300" style={{ borderColor: `${m.color}30` }}>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-bold px-2.5 py-1 rounded-full" style={{ background: `${m.color}15`, color: m.color }}>{m.tag}</span>
                  <span className="text-3xl font-black" style={{ color: m.color }}>{m.acc}%</span>
                </div>
                <h3 className="text-lg font-bold text-slate-800 mb-2">{m.name}</h3>
                <p className="text-sm text-slate-500 mb-4 leading-relaxed">{m.desc}</p>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="bg-slate-50 rounded-lg p-2 text-center">
                    <p className="text-slate-400">Size</p>
                    <p className="font-bold text-slate-700">{m.size}</p>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-2 text-center">
                    <p className="text-slate-400">Inference</p>
                    <p className="font-bold text-slate-700">{m.speed}</p>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-2 text-center">
                    <p className="text-slate-400">Params</p>
                    <p className="font-bold text-slate-700">{m.params}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="text-center mt-8">
            <button onClick={() => onNavigate('comparison')}
              className="px-6 py-3 bg-slate-900 text-white rounded-xl font-semibold text-sm hover:bg-slate-800 transition-all inline-flex items-center gap-2">
              View Full Comparison <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </section>

      {/* Data flow */}
      <section className="py-20 px-4 bg-slate-50">
        <div className="max-w-4xl mx-auto text-center">
          <span className="inline-block bg-emerald-50 text-emerald-700 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider mb-3">Data Flow</span>
          <h2 className="text-3xl font-extrabold text-slate-900 mb-10">End-to-End Pipeline</h2>
          <div className="flex flex-wrap items-center justify-center gap-3 text-sm">
            {[
              { label: 'Patient Input', sub: 'Image + Clinical Data' },
              { label: 'React Frontend', sub: 'Port 3000' },
              { label: 'FastAPI Backend', sub: 'Port 8000' },
              { label: 'Hybrid Model', sub: 'CNN + ANN Fusion' },
              { label: 'Risk Assessment', sub: '5-Class Output' },
            ].map((step, i) => (
              <React.Fragment key={i}>
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm px-5 py-4 min-w-[140px]">
                  <p className="font-bold text-slate-800">{step.label}</p>
                  <p className="text-xs text-slate-400 mt-0.5">{step.sub}</p>
                </div>
                {i < 4 && <ArrowRight className="w-5 h-5 text-indigo-400 flex-shrink-0" />}
              </React.Fragment>
            ))}
          </div>
          <button onClick={() => onNavigate('system')}
            className="mt-8 px-6 py-3 bg-white border border-slate-200 text-slate-700 rounded-xl font-semibold text-sm hover:bg-slate-50 transition-all inline-flex items-center gap-2 shadow-sm">
            View System Architecture <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </section>
    </div>
  )
}
