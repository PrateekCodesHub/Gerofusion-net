import React from 'react'
import { Server, Monitor, Database, ArrowRight, Cpu, Globe, Layers, Code } from 'lucide-react'

export default function SystemArchitecture() {
  return (
    <div className="max-w-6xl mx-auto py-12 px-4 space-y-10 fade-up">
      {/* Header */}
      <div className="text-center mb-4">
        <span className="inline-block bg-emerald-50 text-emerald-700 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider mb-3">Full-Stack Design</span>
        <h1 className="text-4xl font-extrabold text-slate-900 mb-3">System Architecture</h1>
        <p className="text-slate-500 max-w-xl mx-auto">Complete full-stack implementation with React frontend, FastAPI backend, and hybrid deep learning inference engine.</p>
      </div>

      {/* High-level flow */}
      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-8">
        <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2"><Globe className="w-5 h-5 text-indigo-600" /> Data Flow Pipeline</h2>
        <div className="flex flex-wrap items-start justify-center gap-4 mb-8">
          {[
            { icon: Monitor, label: 'User Input', sub: 'Image + 10 Clinical\nFeatures', color: 'bg-blue-50 text-blue-600 border-blue-200' },
            { icon: Globe, label: 'React Frontend', sub: 'Vite + TypeScript\nPort 3000', color: 'bg-indigo-50 text-indigo-600 border-indigo-200' },
            { icon: Server, label: 'FastAPI Backend', sub: 'Python REST API\nPort 8000', color: 'bg-purple-50 text-purple-600 border-purple-200' },
            { icon: Cpu, label: 'Hybrid Model', sub: 'CNN + ANN\nFeature Fusion', color: 'bg-pink-50 text-pink-600 border-pink-200' },
            { icon: Layers, label: 'Classification', sub: '5-Class Softmax\nRisk Scores', color: 'bg-amber-50 text-amber-600 border-amber-200' },
            { icon: Monitor, label: 'Dashboard', sub: 'Visualisation\nGrad-CAM', color: 'bg-emerald-50 text-emerald-600 border-emerald-200' },
          ].map((s, i) => (
            <React.Fragment key={i}>
              <div className={`${s.color} rounded-2xl border p-5 text-center min-w-[130px] transition-all hover:shadow-lg hover:-translate-y-1`}>
                <s.icon className="w-8 h-8 mx-auto mb-2" />
                <p className="font-bold text-sm">{s.label}</p>
                <p className="text-xs mt-1 whitespace-pre-line opacity-75">{s.sub}</p>
              </div>
              {i < 5 && <ArrowRight className="w-5 h-5 text-slate-300 mt-10 flex-shrink-0 hidden md:block" />}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Tech stack grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Frontend */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
          <div className="flex items-center gap-2 mb-5">
            <div className="p-2 bg-blue-50 rounded-xl"><Monitor className="w-5 h-5 text-blue-600" /></div>
            <div>
              <h3 className="font-bold text-slate-800">Frontend — React + Vite</h3>
              <p className="text-xs text-slate-400">Client-side application (Port 3000)</p>
            </div>
          </div>
          <div className="space-y-3">
            {[
              { tech: 'React 18', desc: 'Component-based UI framework with hooks and state management' },
              { tech: 'TypeScript', desc: 'Static type checking for robust development' },
              { tech: 'Vite', desc: 'Next-generation build tool with HMR and optimised bundling' },
              { tech: 'Tailwind CSS', desc: 'Utility-first CSS framework for responsive design' },
              { tech: 'Recharts', desc: 'Composable charting library for training curves, ROC, confusion matrices' },
              { tech: 'Lucide React', desc: 'Icon library for consistent UI elements' },
            ].map(t => (
              <div key={t.tech} className="flex gap-3 items-start p-3 rounded-xl bg-slate-50">
                <Code className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-semibold text-slate-700">{t.tech}</p>
                  <p className="text-xs text-slate-400">{t.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Backend */}
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
          <div className="flex items-center gap-2 mb-5">
            <div className="p-2 bg-purple-50 rounded-xl"><Server className="w-5 h-5 text-purple-600" /></div>
            <div>
              <h3 className="font-bold text-slate-800">Backend — FastAPI + TensorFlow</h3>
              <p className="text-xs text-slate-400">Python inference server (Port 8000)</p>
            </div>
          </div>
          <div className="space-y-3">
            {[
              { tech: 'FastAPI', desc: 'High-performance async web framework with automatic OpenAPI docs' },
              { tech: 'TensorFlow / Keras', desc: 'Deep learning framework for CNN-ANN hybrid model training and inference' },
              { tech: 'NumPy + scikit-learn', desc: 'Numerical computing, preprocessing, and evaluation metrics' },
              { tech: 'Pillow + OpenCV', desc: 'Image preprocessing, resizing, and augmentation pipeline' },
              { tech: 'Matplotlib', desc: 'Grad-CAM heatmap generation and visualisation rendering' },
              { tech: 'Pydantic', desc: 'Data validation and serialisation for API request/response schemas' },
            ].map(t => (
              <div key={t.tech} className="flex gap-3 items-start p-3 rounded-xl bg-slate-50">
                <Code className="w-4 h-4 text-purple-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-semibold text-slate-700">{t.tech}</p>
                  <p className="text-xs text-slate-400">{t.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* API Endpoints */}
      <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
        <div className="flex items-center gap-2 mb-5">
          <div className="p-2 bg-amber-50 rounded-xl"><Server className="w-5 h-5 text-amber-600" /></div>
          <div>
            <h3 className="font-bold text-slate-800">REST API Endpoints</h3>
            <p className="text-xs text-slate-400">Communication interface between frontend and backend</p>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left py-2 text-xs font-bold text-slate-400 uppercase">Method</th>
                <th className="text-left py-2 text-xs font-bold text-slate-400 uppercase">Endpoint</th>
                <th className="text-left py-2 text-xs font-bold text-slate-400 uppercase">Description</th>
                <th className="text-left py-2 text-xs font-bold text-slate-400 uppercase">Response</th>
              </tr>
            </thead>
            <tbody>
              {[
                { method: 'GET', path: '/api/v1/health', desc: 'System health check', resp: 'Status, version, available models' },
                { method: 'GET', path: '/api/v1/models', desc: 'List available model variants', resp: 'Array of model metadata' },
                { method: 'POST', path: '/api/v1/predict', desc: 'Run hybrid CNN-ANN prediction', resp: 'Risk scores, observations, Grad-CAM' },
                { method: 'GET', path: '/api/v1/metrics', desc: 'Model performance metrics', resp: 'Accuracy, F1, confusion matrix, curves' },
                { method: 'GET', path: '/api/v1/metrics/comparison', desc: 'All models side-by-side', resp: 'Combined metrics for comparison' },
                { method: 'GET', path: '/api/v1/architecture/:model', desc: 'Layer-by-layer architecture', resp: 'CNN, ANN, fusion layer details' },
              ].map((e, i) => (
                <tr key={i} className="border-b border-slate-50 hover:bg-slate-50">
                  <td className="py-3">
                    <span className={`text-xs font-bold px-2 py-0.5 rounded ${e.method === 'POST' ? 'bg-amber-100 text-amber-700' : 'bg-emerald-100 text-emerald-700'}`}>
                      {e.method}
                    </span>
                  </td>
                  <td className="py-3 font-mono text-xs text-indigo-600">{e.path}</td>
                  <td className="py-3 text-slate-600">{e.desc}</td>
                  <td className="py-3 text-xs text-slate-400">{e.resp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Database note */}
      <div className="bg-slate-50 rounded-2xl border border-slate-200 p-6">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-5 h-5 text-slate-500" />
          <h3 className="font-bold text-slate-700">Optional: Database Layer</h3>
        </div>
        <p className="text-sm text-slate-500 leading-relaxed">
          For production deployment, a PostgreSQL or MongoDB database can be integrated to persist patient records, 
          assessment history, and longitudinal tracking data. The current prototype operates statelessly — each 
          prediction request is self-contained with no server-side persistence requirement. Database integration 
          would add the following endpoints: <code className="text-xs bg-white px-1.5 py-0.5 rounded border">POST /api/v1/patients</code>, 
          <code className="text-xs bg-white px-1.5 py-0.5 rounded border">GET /api/v1/patients/:id/history</code>, and 
          <code className="text-xs bg-white px-1.5 py-0.5 rounded border">GET /api/v1/analytics/trends</code>.
        </p>
      </div>
    </div>
  )
}
