'use client'

import { useState } from 'react'

const API = 'http://localhost:8000'

// ─────────────────────────────────────────
// HISTORY ITEM COMPONENT
// ─────────────────────────────────────────
function HistoryItem({ item, onClick }) {
  return (
    <button
      onClick={() => onClick(item)}
      className="w-full text-left p-3 rounded-lg border border-gray-200
                 hover:border-blue-300 hover:bg-blue-50 transition-all"
    >
      <div className="flex items-center gap-2 mb-1">
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium
          ${item.language === 'english'
            ? 'bg-blue-100 text-blue-700'
            : 'bg-green-100 text-green-700'
          }`}>
          {item.language === 'english' ? '📖 English' : 'বা Bangla'}
        </span>
        <span className="text-xs text-gray-400">{item.time}</span>
        <span className="text-xs text-gray-400 ml-auto">
          {item.timeTaken}s
        </span>
      </div>
      <p className="text-sm text-gray-500 truncate">
        {item.prompt ? `"${item.prompt}"` : '(no prompt)'}
      </p>
      <p className="text-sm text-gray-700 truncate mt-0.5">
        → {item.result}
      </p>
    </button>
  )
}

// ─────────────────────────────────────────
// SLIDER COMPONENT
// ─────────────────────────────────────────
function Slider({ label, value, min, max, step, onChange, description }) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <label className="text-sm font-medium text-gray-700">
          {label}
        </label>
        <span className="text-sm font-mono bg-gray-100
                         px-2 py-0.5 rounded text-gray-800">
          {value}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-2 bg-gray-200 rounded-lg
                   appearance-none cursor-pointer accent-blue-500"
      />
      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>{min}</span>
        <span className="text-gray-500">{description}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────
export default function Home() {

  // ── state ──
  const [language,    setLanguage]    = useState('english')
  const [prompt,      setPrompt]      = useState('')
  const [temperature, setTemperature] = useState(1.0)
  const [maxTokens,   setMaxTokens]   = useState(200)
  const [topK,        setTopK]        = useState(40)
  const [output,      setOutput]      = useState('')
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState('')
  const [timeTaken,   setTimeTaken]   = useState(null)
  const [history,     setHistory]     = useState([])

  // ── generate ──
  async function handleGenerate() {
    setLoading(true)
    setError('')
    setOutput('')

    try {
      const res = await fetch(`${API}/generate`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          language,
          prompt,
          max_tokens:  maxTokens,
          temperature,
          top_k:       topK,
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'API error')
      }

      const data = await res.json()
      setOutput(data.generated_text)
      setTimeTaken(data.time_taken)

      // add to history
      setHistory(prev => [{
        language,
        prompt,
        result:    data.generated_text,
        timeTaken: data.time_taken,
        time:      new Date().toLocaleTimeString(),
        temperature,
        maxTokens,
      }, ...prev.slice(0, 9)])  // keep last 10

    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  // ── restore from history ──
  function restoreFromHistory(item) {
    setLanguage(item.language)
    setPrompt(item.prompt)
    setTemperature(item.temperature)
    setMaxTokens(item.maxTokens)
    setOutput(item.result)
  }

  // ── temperature description ──
  const tempDesc = temperature < 0.7
    ? 'Conservative'
    : temperature < 1.1
    ? 'Balanced'
    : temperature < 1.4
    ? 'Creative'
    : 'Wild'

  // ─────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────
  return (
<main
  suppressHydrationWarning
  className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 p-4 md:p-8"
>
      <div className="max-w-4xl mx-auto">

        {/* ── Header ── */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🤖 MiniGPT
          </h1>
          <p className="text-gray-500">
            GPT trained from scratch · English & Bangla
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* ── Left Panel: Controls ── */}
          <div className="lg:col-span-1 space-y-4">

            {/* Model Selector */}
            <div className="bg-white rounded-2xl p-5 shadow-sm
                            border border-gray-100">
              <h2 className="text-sm font-semibold text-gray-500
                             uppercase tracking-wide mb-3">
                Model
              </h2>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { key: 'english', label: '📖 English',
                    sub: 'Shakespeare' },
                  { key: 'bangla',  label: 'বা Bangla',
                    sub: 'Wikipedia' },
                ].map(m => (
                  <button
                    key={m.key}
                    onClick={() => {
                      setLanguage(m.key)
                      setPrompt('')
                      setOutput('')
                    }}
                    className={`p-3 rounded-xl border-2 transition-all
                      text-left
                      ${language === m.key
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                      }`}
                  >
                    <div className="font-medium text-sm">{m.label}</div>
                    <div className="text-xs text-gray-400">{m.sub}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Sliders */}
            <div className="bg-white rounded-2xl p-5 shadow-sm
                            border border-gray-100 space-y-5">
              <h2 className="text-sm font-semibold text-gray-500
                             uppercase tracking-wide">
                Settings
              </h2>

              <Slider
                label="Temperature"
                value={temperature}
                min={0.1} max={2.0} step={0.1}
                onChange={setTemperature}
                description={tempDesc}
              />

              <Slider
                label="Max Tokens"
                value={maxTokens}
                min={50} max={500} step={10}
                onChange={setMaxTokens}
                description="length"
              />

              <Slider
                label="Top-K"
                value={topK}
                min={1} max={100} step={1}
                onChange={setTopK}
                description="diversity"
              />
            </div>

            {/* Quick Prompts */}
            <div className="bg-white rounded-2xl p-5 shadow-sm
                            border border-gray-100">
              <h2 className="text-sm font-semibold text-gray-500
                             uppercase tracking-wide mb-3">
                Quick Prompts
              </h2>
              <div className="space-y-2">
                {(language === 'english' ? [
                  'ROMEO:',
                  'To be or not to be',
                  'HAMLET:',
                  'Once upon a time',
                ] : [
                  'বাংলাদেশ একটি',
                  'রবীন্দ্রনাথ ঠাকুর',
                  'ঢাকা শহরে',
                  'মুক্তিযুদ্ধের সময়',
                ]).map(p => (
                  <button
                    key={p}
                    onClick={() => setPrompt(p)}
                    className="w-full text-left text-sm px-3 py-2
                               rounded-lg bg-gray-50 hover:bg-blue-50
                               hover:text-blue-700 transition-all
                               text-gray-700 border border-gray-100
                               hover:border-blue-200"
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* ── Right Panel: Input + Output ── */}
          <div className="lg:col-span-2 space-y-4">

            {/* Prompt Input */}
            <div className="bg-white rounded-2xl p-5 shadow-sm
                            border border-gray-100">
              <h2 className="text-sm font-semibold text-gray-500
                             uppercase tracking-wide mb-3">
                Prompt
                <span className="ml-2 text-xs font-normal
                                 text-gray-400 normal-case">
                  (optional — leave empty to generate freely)
                </span>
              </h2>
              <textarea
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && e.ctrlKey) handleGenerate()
                }}
                placeholder={
                  language === 'english'
                    ? 'ROMEO: or To be or not to be...'
                    : 'বাংলাদেশ একটি...'
                }
                rows={3}
                className="w-full resize-none rounded-xl border
                           border-gray-200 p-3 text-sm
                           focus:outline-none focus:ring-2
                           focus:ring-blue-300 text-gray-800
                           placeholder-gray-300"
              />
              <p className="text-xs text-gray-400 mt-1">
                Ctrl+Enter to generate
              </p>
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={loading}
              className={`w-full py-4 rounded-2xl font-semibold
                         text-white text-lg transition-all shadow-sm
                ${loading
                  ? 'bg-blue-300 cursor-not-allowed'
                  : 'bg-blue-500 hover:bg-blue-600 active:scale-95'
                }`}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5"
                       viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12"
                            r="10" stroke="currentColor"
                            strokeWidth="4"/>
                    <path className="opacity-75" fill="currentColor"
                          d="M4 12a8 8 0 018-8v8H4z"/>
                  </svg>
                  Generating...
                </span>
              ) : (
                '✨ Generate Text'
              )}
            </button>

            {/* Error */}
            {error && (
              <div className="bg-red-50 border border-red-200
                              rounded-2xl p-4 text-red-700 text-sm">
                ⚠️ {error}
              </div>
            )}

            {/* Output */}
            {output && (
              <div className="bg-white rounded-2xl p-5 shadow-sm
                              border border-gray-100">
                <div className="flex justify-between items-center mb-3">
                  <h2 className="text-sm font-semibold text-gray-500
                                 uppercase tracking-wide">
                    Generated Text
                  </h2>
                  <div className="flex items-center gap-3">
                    {timeTaken && (
                      <span className="text-xs text-gray-400">
                        ⏱ {timeTaken}s
                      </span>
                    )}
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(
                          prompt + output
                        )
                      }}
                      className="text-xs text-blue-500
                                 hover:text-blue-700 transition-colors"
                    >
                      Copy
                    </button>
                  </div>
                </div>

                {/* full text = prompt + output */}
                <div className="bg-gray-50 rounded-xl p-4
                                text-sm text-gray-800 leading-relaxed
                                whitespace-pre-wrap font-mono
                                max-h-80 overflow-y-auto">
                  {prompt && (
                    <span className="text-blue-600 font-semibold">
                      {prompt}
                    </span>
                  )}
                  {output}
                </div>
              </div>
            )}

            {/* History */}
            {history.length > 0 && (
              <div className="bg-white rounded-2xl p-5 shadow-sm
                              border border-gray-100">
                <div className="flex justify-between items-center mb-3">
                  <h2 className="text-sm font-semibold text-gray-500
                                 uppercase tracking-wide">
                    History
                  </h2>
                  <button
                    onClick={() => setHistory([])}
                    className="text-xs text-red-400
                               hover:text-red-600 transition-colors"
                  >
                    Clear
                  </button>
                </div>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {history.map((item, i) => (
                    <HistoryItem
                      key={i}
                      item={item}
                      onClick={restoreFromHistory}
                    />
                  ))}
                </div>
              </div>
            )}

          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-xs text-gray-400">
          Built from scratch · PyTorch · FastAPI · Next.js
        </div>

      </div>
    </main>
  )
}