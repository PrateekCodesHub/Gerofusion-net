import React, { useState } from 'react'
import LandingHero from './components/LandingHero'
import AnalysisWorkflow from './components/AnalysisWorkflow'
import Dashboard from './components/Dashboard'
import ModelComparison from './components/ModelComparison'
import ArchitectureExplorer from './components/ArchitectureExplorer'
import SystemArchitecture from './components/SystemArchitecture'
import { PhenotypingReport, PageView } from './types'
import { Brain, Stethoscope, Home, BarChart2, Layers, Server } from 'lucide-react'

export default function App() {
  const [page, setPage] = useState<PageView>('home')
  const [report, setReport] = useState<PhenotypingReport | null>(null)

  const handleAnalysisComplete = (r: PhenotypingReport) => {
    setReport(r)
    setPage('dashboard')
  }

  const handleReset = () => {
    setReport(null)
    setPage('analysis')
  }

  const navItems: { id: PageView; label: string; icon: React.ReactNode }[] = [
    { id: 'home', label: 'Home', icon: <Home className="w-4 h-4" /> },
    { id: 'analysis', label: 'Analysis', icon: <Stethoscope className="w-4 h-4" /> },
    { id: 'comparison', label: 'Comparison', icon: <BarChart2 className="w-4 h-4" /> },
    { id: 'architecture', label: 'Architecture', icon: <Layers className="w-4 h-4" /> },
    { id: 'system', label: 'System', icon: <Server className="w-4 h-4" /> },
  ]

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-slate-100 px-4 md:px-8 py-3">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <button className="flex items-center gap-2" onClick={() => setPage('home')}>
            <div className="bg-indigo-600 p-2 rounded-xl"><Brain className="w-5 h-5 text-white" /></div>
            <span className="text-lg font-black text-slate-900 tracking-tight">GeroFusion</span>
            <span className="hidden sm:inline text-xs bg-indigo-50 text-indigo-600 font-bold px-2 py-0.5 rounded-full">Hybrid CNN-ANN</span>
          </button>
          <div className="flex items-center gap-1">
            {navItems.map(item => (
              <button key={item.id} onClick={() => setPage(item.id)}
                className={`hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  page === item.id
                    ? 'bg-indigo-50 text-indigo-700'
                    : 'text-slate-400 hover:text-slate-600 hover:bg-slate-50'
                }`}>
                {item.icon}
                {item.label}
              </button>
            ))}
            {report && page !== 'dashboard' && (
              <button onClick={() => setPage('dashboard')}
                className="px-3 py-1.5 bg-indigo-600 text-white rounded-lg font-semibold text-sm hover:bg-indigo-700 transition-colors ml-2">
                View Report
              </button>
            )}
          </div>
        </div>
      </nav>

      <main className="flex-grow">
        {page === 'home' && <LandingHero onNavigate={setPage} />}
        {page === 'analysis' && <AnalysisWorkflow onComplete={handleAnalysisComplete} />}
        {page === 'dashboard' && report && <Dashboard report={report} onReset={handleReset} />}
        {page === 'comparison' && <ModelComparison />}
        {page === 'architecture' && <ArchitectureExplorer />}
        {page === 'system' && <SystemArchitecture />}
      </main>

      <footer className="bg-white border-t border-slate-100 py-8 px-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4 text-sm text-slate-400">
          <div className="flex items-center gap-2">
            <Stethoscope className="w-4 h-4 text-indigo-500" />
            <span className="font-semibold text-slate-600">Geriatric Syndrome GeroFusion · Hybrid CNN-ANN Architecture</span>
          </div>
          <p>Multi-modal deep learning framework for geriatric syndrome detection. 
            Made By: Prateek Vasnik - URK23EC4031
                  
          </p>
        </div>
      </footer>
    </div>
  )
}
