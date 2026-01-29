import { useState, useEffect } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  AreaChart,
  Area,
  ComposedChart,
  ReferenceLine
} from 'recharts'
import './App.css'

const API_BASE = 'http://127.0.0.1:8002'

const MODEL_COLORS = {
  xgb: '#3b82f6',
  rf: '#10b981',
  linear: '#f59e0b'
}

const MODEL_NAMES = {
  xgb: 'XGBoost',
  rf: 'Random Forest',
  linear: 'Linear Regression'
}

function App() {
  const [granularities, setGranularities] = useState([])
  const [available, setAvailable] = useState({})
  const [selectedGranularity, setSelectedGranularity] = useState('D')
  const [allModelsData, setAllModelsData] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedModel, setSelectedModel] = useState('xgb')

  // Chat state
  const [chatOpen, setChatOpen] = useState(false)
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)

  // Fetch available granularities and models on mount
  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/granularities`).then(r => r.json()),
      fetch(`${API_BASE}/available`).then(r => r.json())
    ])
      .then(([granData, availData]) => {
        setGranularities(granData.granularities)
        setAvailable(availData.available)
      })
      .catch(err => setError('Failed to connect to API. Make sure the server is running on port 8000.'))
  }, [])

  // Fetch all models data when granularity changes
  useEffect(() => {
    if (!selectedGranularity || !available[selectedGranularity]) return

    const models = available[selectedGranularity]
    if (models.length === 0) {
      setAllModelsData({})
      setLoading(false)
      return
    }

    setLoading(true)
    setError(null)

    const fetches = models.map(m =>
      Promise.all([
        fetch(`${API_BASE}/metrics?granularity=${selectedGranularity}&model=${m.model}&horizon=${m.horizon}`)
          .then(r => r.ok ? r.json() : null),
        fetch(`${API_BASE}/predict?granularity=${selectedGranularity}&model=${m.model}&horizon=${m.horizon}`)
          .then(r => r.ok ? r.json() : null)
      ]).then(([metrics, preds]) => ({ model: m.model, horizon: m.horizon, metrics, preds }))
    )

    Promise.all(fetches)
      .then(results => {
        const data = {}
        results.forEach(r => {
          if (r.metrics && r.preds) {
            data[r.model] = r
          }
        })
        setAllModelsData(data)
        if (!data[selectedModel] && Object.keys(data).length > 0) {
          setSelectedModel(Object.keys(data)[0])
        }
        setLoading(false)
      })
      .catch(err => {
        setError('Failed to fetch model data')
        setLoading(false)
      })
  }, [selectedGranularity, available])

  const models = Object.keys(allModelsData)
  const currentData = allModelsData[selectedModel]
  const granConfig = granularities.find(g => g.code === selectedGranularity)

  // Compute comparison data for all models
  const getComparisonData = () => {
    if (models.length === 0) return []

    const firstModel = allModelsData[models[0]]
    if (!firstModel?.preds?.series) return []

    return firstModel.preds.series.map((item, idx) => {
      const point = {
        time: formatTime(item.t, selectedGranularity),
        actual: item.actual
      }
      models.forEach(m => {
        const pred = allModelsData[m]?.preds?.series?.[idx]
        if (pred) {
          point[m] = pred.predicted
          point[`${m}_error`] = pred.predicted - item.actual
        }
      })
      return point
    })
  }

  const comparisonData = getComparisonData()

  // Best model by metric
  const getBestModel = (metric) => {
    let best = null
    let bestVal = Infinity
    models.forEach(m => {
      const val = allModelsData[m]?.metrics?.[metric]
      if (val !== undefined && val < bestVal) {
        bestVal = val
        best = m
      }
    })
    return best
  }

  // Send chat message
  const sendChatMessage = async () => {
    if (!chatInput.trim() || chatLoading) return

    const userMessage = chatInput
    setChatInput('')
    setChatMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setChatLoading(true)

    try {
      // Include current view context
      const context = {
        granularity: selectedGranularity,
        selectedModel: selectedModel,
        metrics: allModelsData[selectedModel]?.metrics,
        activeTab: activeTab
      }

      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, context })
      })

      const data = await res.json()
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }])
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }])
    } finally {
      setChatLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Electricity Demand Forecasting</h1>
        <p className="subtitle">Multi-timeframe ML model comparison dashboard</p>
      </header>

      {error && <div className="error">{error}</div>}

      <div className="controls-bar">
        <div className="granularity-tabs">
          {granularities.map(g => (
            <button
              key={g.code}
              className={`gran-tab ${selectedGranularity === g.code ? 'active' : ''}`}
              onClick={() => setSelectedGranularity(g.code)}
            >
              {g.name.charAt(0).toUpperCase() + g.name.slice(1)}
              <span className="tab-count">{available[g.code]?.length || 0} models</span>
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="loading">
          <div className="spinner"></div>
          Loading forecast data...
        </div>
      ) : models.length === 0 ? (
        <div className="no-data">
          <h2>No models trained for {granConfig?.name} forecasting</h2>
          <p>Run <code>python ml/train_all.py --granularities {selectedGranularity}</code> to train models.</p>
        </div>
      ) : (
        <>
          {/* Summary Cards */}
          <div className="summary-section">
            <h2>Model Performance Summary</h2>
            <div className="model-cards">
              {models.map(m => {
                const data = allModelsData[m]
                const isBest = getBestModel('smape') === m
                return (
                  <div
                    key={m}
                    className={`model-card ${selectedModel === m ? 'selected' : ''} ${isBest ? 'best' : ''}`}
                    onClick={() => setSelectedModel(m)}
                    style={{ borderColor: MODEL_COLORS[m] }}
                  >
                    {isBest && <div className="best-badge">Best</div>}
                    <div className="model-name" style={{ color: MODEL_COLORS[m] }}>
                      {MODEL_NAMES[m]}
                    </div>
                    <div className="model-metrics">
                      <div className="mini-metric">
                        <span className="label">SMAPE</span>
                        <span className="value">{data.metrics.smape?.toFixed(2)}%</span>
                      </div>
                      <div className="mini-metric">
                        <span className="label">MAE</span>
                        <span className="value">{formatNumber(data.metrics.mae)}</span>
                      </div>
                      <div className="mini-metric">
                        <span className="label">RMSE</span>
                        <span className="value">{formatNumber(data.metrics.rmse)}</span>
                      </div>
                    </div>
                    <div className="model-horizon">
                      {data.horizon} {getHorizonUnit(selectedGranularity)} forecast
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="nav-tabs">
            <button
              className={activeTab === 'overview' ? 'active' : ''}
              onClick={() => setActiveTab('overview')}
            >
              Overview
            </button>
            <button
              className={activeTab === 'comparison' ? 'active' : ''}
              onClick={() => setActiveTab('comparison')}
            >
              Model Comparison
            </button>
            <button
              className={activeTab === 'details' ? 'active' : ''}
              onClick={() => setActiveTab('details')}
            >
              Detailed Analysis
            </button>
          </div>

          {/* Overview Tab */}
          {activeTab === 'overview' && currentData && (
            <div className="tab-content">
              <div className="chart-section">
                <h3>Forecast vs Actual - {MODEL_NAMES[selectedModel]}</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={comparisonData}>
                    <defs>
                      <linearGradient id="actualGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="predGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={MODEL_COLORS[selectedModel]} stopOpacity={0.3}/>
                        <stop offset="95%" stopColor={MODEL_COLORS[selectedModel]} stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} tickFormatter={formatNumber} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Area type="monotone" dataKey="actual" stroke="#6366f1" fill="url(#actualGrad)" strokeWidth={2} name="Actual" />
                    <Area type="monotone" dataKey={selectedModel} stroke={MODEL_COLORS[selectedModel]} fill="url(#predGrad)" strokeWidth={2} name="Predicted" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div className="metrics-detail">
                <h3>Performance Metrics</h3>
                <div className="metrics-grid">
                  <MetricCard
                    label="MAE"
                    value={formatNumber(currentData.metrics.mae)}
                    unit="MW"
                    description="Average absolute error"
                    color={MODEL_COLORS[selectedModel]}
                  />
                  <MetricCard
                    label="RMSE"
                    value={formatNumber(currentData.metrics.rmse)}
                    unit="MW"
                    description="Root mean square error"
                    color={MODEL_COLORS[selectedModel]}
                  />
                  <MetricCard
                    label="SMAPE"
                    value={currentData.metrics.smape?.toFixed(2)}
                    unit="%"
                    description="Symmetric MAPE"
                    color={MODEL_COLORS[selectedModel]}
                  />
                  <MetricCard
                    label="MAPE"
                    value={currentData.metrics.mape?.toFixed(2)}
                    unit="%"
                    description="Mean absolute % error"
                    color={MODEL_COLORS[selectedModel]}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Comparison Tab */}
          {activeTab === 'comparison' && (
            <div className="tab-content">
              <div className="chart-section">
                <h3>All Models Comparison</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} tickFormatter={formatNumber} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Line type="monotone" dataKey="actual" stroke="#6366f1" strokeWidth={3} dot={false} name="Actual" />
                    {models.map(m => (
                      <Line
                        key={m}
                        type="monotone"
                        dataKey={m}
                        stroke={MODEL_COLORS[m]}
                        strokeWidth={2}
                        dot={false}
                        strokeDasharray={m === 'linear' ? '5 5' : undefined}
                        name={MODEL_NAMES[m]}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-section">
                <h3>Error Comparison</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} tickFormatter={formatNumber} />
                    <Tooltip content={<CustomTooltip isError />} />
                    <Legend />
                    <ReferenceLine y={0} stroke="#64748b" />
                    {models.map(m => (
                      <Bar key={m} dataKey={`${m}_error`} fill={MODEL_COLORS[m]} name={`${MODEL_NAMES[m]} Error`} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="comparison-table">
                <h3>Metrics Comparison</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>MAE (MW)</th>
                      <th>RMSE (MW)</th>
                      <th>SMAPE (%)</th>
                      <th>MAPE (%)</th>
                      <th>Rank</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models
                      .map(m => ({ model: m, ...allModelsData[m].metrics }))
                      .sort((a, b) => a.smape - b.smape)
                      .map((row, idx) => (
                        <tr key={row.model} className={idx === 0 ? 'best-row' : ''}>
                          <td style={{ color: MODEL_COLORS[row.model], fontWeight: 600 }}>
                            {MODEL_NAMES[row.model]}
                          </td>
                          <td>{formatNumber(row.mae)}</td>
                          <td>{formatNumber(row.rmse)}</td>
                          <td>{row.smape?.toFixed(2)}</td>
                          <td>{row.mape?.toFixed(2)}</td>
                          <td className="rank">#{idx + 1}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Details Tab */}
          {activeTab === 'details' && currentData && (
            <div className="tab-content">
              <div className="details-header">
                <h3>{MODEL_NAMES[selectedModel]} - Prediction Details</h3>
                <div className="model-selector">
                  {models.map(m => (
                    <button
                      key={m}
                      className={selectedModel === m ? 'active' : ''}
                      style={{ borderColor: MODEL_COLORS[m], color: selectedModel === m ? '#fff' : MODEL_COLORS[m] }}
                      onClick={() => setSelectedModel(m)}
                    >
                      {MODEL_NAMES[m]}
                    </button>
                  ))}
                </div>
              </div>

              <div className="stats-row">
                <StatBox
                  label="Total Predictions"
                  value={currentData.preds.series.length}
                />
                <StatBox
                  label="Avg Actual"
                  value={formatNumber(avg(currentData.preds.series.map(s => s.actual)))}
                  unit="MW"
                />
                <StatBox
                  label="Avg Predicted"
                  value={formatNumber(avg(currentData.preds.series.map(s => s.predicted)))}
                  unit="MW"
                />
                <StatBox
                  label="Max Error"
                  value={formatNumber(Math.max(...currentData.preds.series.map(s => Math.abs(s.predicted - s.actual))))}
                  unit="MW"
                />
              </div>

              <div className="data-table">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Actual (MW)</th>
                      <th>Predicted (MW)</th>
                      <th>Error (MW)</th>
                      <th>Error %</th>
                      <th>Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentData.preds.series.map((item, i) => {
                      const error = item.predicted - item.actual
                      const errorPct = (error / item.actual) * 100
                      const accuracy = 100 - Math.abs(errorPct)
                      return (
                        <tr key={i}>
                          <td className="time-cell">{formatTime(item.t, selectedGranularity)}</td>
                          <td>{formatNumber(item.actual)}</td>
                          <td>{formatNumber(item.predicted)}</td>
                          <td className={error > 0 ? 'error-pos' : 'error-neg'}>
                            {error > 0 ? '+' : ''}{formatNumber(error)}
                          </td>
                          <td className={error > 0 ? 'error-pos' : 'error-neg'}>
                            {error > 0 ? '+' : ''}{errorPct.toFixed(2)}%
                          </td>
                          <td>
                            <div className="accuracy-bar">
                              <div
                                className="accuracy-fill"
                                style={{
                                  width: `${Math.max(0, accuracy)}%`,
                                  backgroundColor: accuracy > 95 ? '#10b981' : accuracy > 90 ? '#f59e0b' : '#ef4444'
                                }}
                              />
                              <span>{accuracy.toFixed(1)}%</span>
                            </div>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* Chat Widget */}
      <ChatWidget
        open={chatOpen}
        onToggle={() => setChatOpen(!chatOpen)}
        messages={chatMessages}
        input={chatInput}
        onInputChange={setChatInput}
        onSend={sendChatMessage}
        loading={chatLoading}
      />
    </div>
  )
}

function MetricCard({ label, value, unit, description, color }) {
  return (
    <div className="metric-card" style={{ borderTopColor: color }}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {value} <span className="metric-unit">{unit}</span>
      </div>
      <div className="metric-description">{description}</div>
    </div>
  )
}

function StatBox({ label, value, unit }) {
  return (
    <div className="stat-box">
      <div className="stat-value">{value} {unit && <span className="stat-unit">{unit}</span>}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

function CustomTooltip({ active, payload, label, isError }) {
  if (!active || !payload) return null
  return (
    <div className="custom-tooltip">
      <div className="tooltip-label">{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="tooltip-row" style={{ color: p.color }}>
          <span className="tooltip-name">{p.name}:</span>
          <span className="tooltip-value">
            {isError ? (p.value > 0 ? '+' : '') : ''}{formatNumber(p.value)} MW
          </span>
        </div>
      ))}
    </div>
  )
}

function ChatWidget({ open, onToggle, messages, input, onInputChange, onSend, loading }) {
  return (
    <div className="chat-widget">
      {/* Toggle button */}
      <button className="chat-toggle" onClick={onToggle}>
        {open ? '\u2715' : '\uD83D\uDCAC'}
      </button>

      {/* Chat panel */}
      {open && (
        <div className="chat-panel">
          <div className="chat-header">
            <span>AI Assistant</span>
          </div>

          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="chat-welcome">
                Ask me about the electricity demand forecasts, model performance, or energy topics!
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`chat-message ${msg.role}`}>
                {msg.content}
              </div>
            ))}
            {loading && <div className="chat-message assistant typing">Thinking...</div>}
          </div>

          <div className="chat-input-area">
            <input
              type="text"
              value={input}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && onSend()}
              placeholder="Ask about the data..."
            />
            <button onClick={onSend} disabled={loading}>Send</button>
          </div>
        </div>
      )}
    </div>
  )
}

function getHorizonUnit(granularity) {
  const units = { H: 'hour', D: 'day', W: 'week', M: 'month', Y: 'year' }
  return units[granularity] || 'period'
}

function formatTime(isoString, granularity) {
  const date = new Date(isoString)
  switch (granularity) {
    case 'H': return date.toLocaleString('en-GB', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    case 'D': return date.toLocaleDateString('en-GB', { weekday: 'short', month: 'short', day: 'numeric' })
    case 'W': return `Week ${getWeekNumber(date)}`
    case 'M': return date.toLocaleDateString('en-GB', { month: 'short', year: 'numeric' })
    case 'Y': return date.getFullYear().toString()
    default: return isoString
  }
}

function getWeekNumber(date) {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()))
  const dayNum = d.getUTCDay() || 7
  d.setUTCDate(d.getUTCDate() + 4 - dayNum)
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1))
  return Math.ceil((((d - yearStart) / 86400000) + 1) / 7)
}

function formatNumber(num) {
  if (num === undefined || num === null) return '-'
  if (Math.abs(num) >= 1000000) return (num / 1000000).toFixed(2) + 'M'
  if (Math.abs(num) >= 1000) return (num / 1000).toFixed(1) + 'k'
  return Math.round(num).toLocaleString()
}

function avg(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length
}

export default App
