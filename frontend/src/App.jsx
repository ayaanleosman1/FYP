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

const API_BASE = 'http://127.0.0.1:8000'

const MODEL_COLORS = {
  xgb: '#3b82f6',
  rf: '#10b981',
  linear: '#f59e0b',
  ebm: '#8b5cf6'
}

const MODEL_NAMES = {
  xgb: 'XGBoost',
  rf: 'Random Forest',
  linear: 'Linear Regression',
  ebm: 'Explainable Boosting Machine'
}

const DATA_INFO = {
  source: 'UK National Grid (NESO)',
  years: '2009-2024',
  totalHours: '140,240',
  features: 'Demand + Weather (temperature, humidity, wind)',
  url: 'https://www.neso.energy/data-portal/historic-demand-data'
}

function App() {
  const [granularities, setGranularities] = useState([])
  const [available, setAvailable] = useState({})
  const [selectedGranularity, setSelectedGranularity] = useState('D')
  const [allModelsData, setAllModelsData] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('forecast')
  const [forecastView, setForecastView] = useState('overview')
  const [selectedModel, setSelectedModel] = useState('xgb')
  // Chat state
  const [chatOpen, setChatOpen] = useState(false)
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)

  // What-If state
  const [whatIfFeatures, setWhatIfFeatures] = useState(null)
  const [whatIfValues, setWhatIfValues] = useState({})
  const [whatIfPrediction, setWhatIfPrediction] = useState(null)
  const [whatIfLoading, setWhatIfLoading] = useState(false)
  const [whatIfBaseline, setWhatIfBaseline] = useState(null)

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
        timestamp: item.t,
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

  // Fetch What-If features when tab is selected
  const loadWhatIfFeatures = async () => {
    try {
      const res = await fetch(`${API_BASE}/whatif/features?granularity=H&horizon=24`)
      if (res.ok) {
        const data = await res.json()
        setWhatIfFeatures(data)
        // Initialize with median values
        const initialValues = {}
        Object.entries(data.feature_ranges).forEach(([feat, range]) => {
          initialValues[feat] = range.median
        })
        setWhatIfValues(initialValues)
        // Get initial prediction and save as baseline
        fetchWhatIfPrediction(initialValues, true)
      }
    } catch (err) {
      console.error('Failed to load what-if features:', err)
    }
  }

  // Fetch What-If prediction
  const fetchWhatIfPrediction = async (values, isBaseline = false) => {
    setWhatIfLoading(true)
    try {
      const res = await fetch(`${API_BASE}/whatif`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: values,
          granularity: 'H',
          horizon: 24
        })
      })
      if (res.ok) {
        const data = await res.json()
        setWhatIfPrediction(data)
        if (isBaseline) {
          setWhatIfBaseline(data.prediction)
        }
      }
    } catch (err) {
      console.error('Failed to get prediction:', err)
    } finally {
      setWhatIfLoading(false)
    }
  }

  // Handle slider change
  const handleWhatIfChange = (feature, value) => {
    const newValues = { ...whatIfValues, [feature]: parseFloat(value) }
    setWhatIfValues(newValues)
    fetchWhatIfPrediction(newValues)
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
      {/* Navigation Bar */}
      <nav className="navbar">
        <div className="nav-brand">
          <span className="nav-logo">&#9889;</span>
          <span className="nav-title">UK Demand Forecast</span>
        </div>
        <div className="nav-links">
          <button
            className={`nav-link ${activeTab === 'forecast' ? 'active' : ''}`}
            onClick={() => setActiveTab('forecast')}
          >
            Forecast
          </button>
          <button
            className={`nav-link ${activeTab === 'whatif' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('whatif')
              if (!whatIfFeatures) loadWhatIfFeatures()
            }}
          >
            What-If Analysis
          </button>
          <button
            className={`nav-link chat-link ${chatOpen ? 'active' : ''}`}
            onClick={() => setChatOpen(!chatOpen)}
          >
            <span className="chat-icon">&#128172;</span>
            AI Assistant
          </button>
        </div>
      </nav>

      <header className="header">
        <h1>UK Electricity Demand Forecasting</h1>
        <p className="subtitle">ML models trained on 16 years of real National Grid data</p>
        <div className="data-badge">
          <span className="badge-item">
            <span className="badge-icon">&#9889;</span>
            {DATA_INFO.source}
          </span>
          <span className="badge-item">
            <span className="badge-icon">&#128197;</span>
            {DATA_INFO.years}
          </span>
          <span className="badge-item">
            <span className="badge-icon">&#127777;</span>
            Weather Features
          </span>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      {/* Forecast Section */}
      {activeTab === 'forecast' && (
        <>
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

          {/* Forecast Sub-Navigation */}
          <div className="nav-tabs">
            <button
              className={forecastView === 'overview' ? 'active' : ''}
              onClick={() => setForecastView('overview')}
            >
              Overview
            </button>
            <button
              className={forecastView === 'comparison' ? 'active' : ''}
              onClick={() => setForecastView('comparison')}
            >
              Model Comparison
            </button>
            <button
              className={forecastView === 'details' ? 'active' : ''}
              onClick={() => setForecastView('details')}
            >
              Detailed Analysis
            </button>
          </div>

          {/* Overview Tab */}
          {forecastView === 'overview' && currentData && (
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
          {forecastView === 'comparison' && (
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
          {forecastView === 'details' && currentData && (
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
      </>
      )}

      {/* What-If Section */}
      {activeTab === 'whatif' && (
            <div className="tab-content whatif-tab">
              <div className="whatif-header">
                <h3>What-If Scenario Analysis</h3>
                <p className="whatif-description">
                  Explore how different conditions affect UK electricity demand. Model trained on 16 years of real National Grid data (2009-2024).
                </p>
              </div>

              {!whatIfFeatures ? (
                <div className="loading">Loading features...</div>
              ) : (
                <>
                  {/* Preset Scenarios */}
                  <div className="scenario-presets">
                    <h4>Quick Scenarios</h4>
                    <div className="preset-grid">
                      <button
                        className="preset-card"
                        onClick={() => {
                          // Cold winter morning: high demand due to heating + morning ramp up
                          const vals = {
                            hour: 8, dow: 1, month: 1, temp: 2, humidity: 85, wind_speed: 15,
                            lag_1: 38000, lag_24: 35000, lag_168: 37000, roll_24_mean: 34000
                          }
                          setWhatIfValues(vals)
                          fetchWhatIfPrediction(vals)
                        }}
                      >
                        <span className="preset-icon">🌅</span>
                        <span className="preset-name">Cold Winter Morning</span>
                        <span className="preset-desc">8 AM, Monday, 2°C</span>
                      </button>
                      <button
                        className="preset-card"
                        onClick={() => {
                          // Summer evening peak: moderate demand, evening activities
                          const vals = {
                            hour: 18, dow: 2, month: 7, temp: 24, humidity: 55, wind_speed: 10,
                            lag_1: 30000, lag_24: 28000, lag_168: 29000, roll_24_mean: 27000
                          }
                          setWhatIfValues(vals)
                          fetchWhatIfPrediction(vals)
                        }}
                      >
                        <span className="preset-icon">🌇</span>
                        <span className="preset-name">Summer Evening Peak</span>
                        <span className="preset-desc">6 PM, Weekday, 24°C</span>
                      </button>
                      <button
                        className="preset-card"
                        onClick={() => {
                          // Weekend night: lowest demand - everyone sleeping
                          const vals = {
                            hour: 3, dow: 6, month: 5, temp: 12, humidity: 75, wind_speed: 8,
                            lag_1: 18000, lag_24: 20000, lag_168: 18500, roll_24_mean: 22000
                          }
                          setWhatIfValues(vals)
                          fetchWhatIfPrediction(vals)
                        }}
                      >
                        <span className="preset-icon">🌙</span>
                        <span className="preset-name">Weekend Night</span>
                        <span className="preset-desc">3 AM, Saturday, 12°C</span>
                      </button>
                      <button
                        className="preset-card"
                        onClick={() => {
                          // Peak winter evening: highest demand - cold + everyone home
                          const vals = {
                            hour: 17, dow: 3, month: 12, temp: -2, humidity: 90, wind_speed: 20,
                            lag_1: 42000, lag_24: 40000, lag_168: 41000, roll_24_mean: 38000
                          }
                          setWhatIfValues(vals)
                          fetchWhatIfPrediction(vals)
                        }}
                      >
                        <span className="preset-icon">❄️</span>
                        <span className="preset-name">Peak Winter Evening</span>
                        <span className="preset-desc">5 PM, December, -2°C</span>
                      </button>
                    </div>
                  </div>

                  <div className="whatif-container">
                    {/* Result Panel - Now on left for prominence */}
                    <div className="whatif-result">
                      <div className="result-header">
                        <h4>Predicted Demand</h4>
                        {whatIfBaseline && (
                          <span className="baseline-label">Baseline: {formatNumber(whatIfBaseline)} MW</span>
                        )}
                      </div>

                      {whatIfLoading ? (
                        <div className="prediction-loading">
                          <div className="pulse-ring"></div>
                          Calculating...
                        </div>
                      ) : whatIfPrediction ? (
                        <>
                          {/* Visual Gauge */}
                          <div className="demand-gauge">
                            <svg viewBox="0 0 200 120" className="gauge-svg">
                              <defs>
                                <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                  <stop offset="0%" stopColor="#10b981" />
                                  <stop offset="50%" stopColor="#f59e0b" />
                                  <stop offset="100%" stopColor="#ef4444" />
                                </linearGradient>
                              </defs>
                              {/* Background arc */}
                              <path
                                d="M 20 100 A 80 80 0 0 1 180 100"
                                fill="none"
                                stroke="#334155"
                                strokeWidth="12"
                                strokeLinecap="round"
                              />
                              {/* Filled arc based on prediction */}
                              <path
                                d="M 20 100 A 80 80 0 0 1 180 100"
                                fill="none"
                                stroke="url(#gaugeGradient)"
                                strokeWidth="12"
                                strokeLinecap="round"
                                strokeDasharray={`${((whatIfPrediction.prediction - 15000) / 30000) * 251} 251`}
                              />
                              {/* Needle */}
                              <line
                                x1="100"
                                y1="100"
                                x2={100 + 60 * Math.cos(Math.PI - ((whatIfPrediction.prediction - 15000) / 30000) * Math.PI)}
                                y2={100 - 60 * Math.sin(Math.PI - ((whatIfPrediction.prediction - 15000) / 30000) * Math.PI)}
                                stroke="#f1f5f9"
                                strokeWidth="3"
                                strokeLinecap="round"
                              />
                              <circle cx="100" cy="100" r="8" fill="#f1f5f9" />
                            </svg>
                            <div className="gauge-labels">
                              <span>15k</span>
                              <span>30k</span>
                              <span>45k</span>
                            </div>
                          </div>

                          <div className="prediction-value">
                            {Math.round(whatIfPrediction.prediction).toLocaleString()}
                            <span className="prediction-unit">MW</span>
                          </div>

                          {whatIfBaseline && whatIfPrediction.prediction !== whatIfBaseline && (
                            <div className={`prediction-change ${whatIfPrediction.prediction > whatIfBaseline ? 'increase' : 'decrease'}`}>
                              {whatIfPrediction.prediction > whatIfBaseline ? '▲' : '▼'}
                              {' '}{Math.abs(whatIfPrediction.prediction - whatIfBaseline).toFixed(0)} MW
                              {' '}({((whatIfPrediction.prediction - whatIfBaseline) / whatIfBaseline * 100).toFixed(2)}%)
                            </div>
                          )}

                          {/* Impact Summary */}
                          <div className="impact-summary">
                            {Object.entries(whatIfPrediction.contributions || {})
                              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                              .slice(0, 3)
                              .map(([feat, value]) => (
                                <div key={feat} className={`impact-chip ${value >= 0 ? 'positive' : 'negative'}`}>
                                  {formatFeatureName(feat)}: {value >= 0 ? '+' : ''}{value.toFixed(0)}
                                </div>
                              ))}
                          </div>
                        </>
                      ) : null}
                    </div>

                    {/* Sliders Panel */}
                    <div className="whatif-sliders">
                      <div className="sliders-header">
                        <h4>Adjust Conditions</h4>
                        <button
                          className="reset-btn"
                          onClick={() => {
                            const resetValues = {}
                            Object.entries(whatIfFeatures.feature_ranges).forEach(([feat, range]) => {
                              resetValues[feat] = range.median
                            })
                            setWhatIfValues(resetValues)
                            fetchWhatIfPrediction(resetValues)
                          }}
                        >
                          Reset
                        </button>
                      </div>

                      {/* Key controllable features - including lag features which have highest impact */}
                      {['lag_1', 'hour', 'lag_24', 'temp', 'dow', 'humidity', 'wind_speed', 'month'].map(feat => {
                        const range = whatIfFeatures.feature_ranges[feat]
                        if (!range) return null
                        const importance = whatIfFeatures.feature_importances[feat] || 0
                        const maxImportance = Math.max(...Object.values(whatIfFeatures.feature_importances))

                        return (
                          <div key={feat} className="whatif-slider">
                            <div className="slider-header">
                              <label>
                                <span className="feature-icon">{getFeatureIcon(feat)}</span>
                                {formatFeatureName(feat)}
                              </label>
                              <span className="slider-value">{formatFeatureValue(feat, whatIfValues[feat])}</span>
                            </div>
                            <div className="slider-track-container">
                              <input
                                type="range"
                                min={range.min}
                                max={range.max}
                                step={feat === 'hour' || feat === 'dow' || feat === 'month' ? 1 : (range.max - range.min) / 100}
                                value={whatIfValues[feat] || range.median}
                                onChange={(e) => handleWhatIfChange(feat, e.target.value)}
                              />
                              <div
                                className="importance-indicator"
                                style={{ opacity: 0.3 + (importance / maxImportance) * 0.7 }}
                                title={`Impact: ${(importance / maxImportance * 100).toFixed(0)}%`}
                              />
                            </div>
                          </div>
                        )
                      })}

                      <div className="slider-note">
                        <p><strong>Why "Previous Hour Demand" matters most:</strong> Electricity demand is highly predictable - if 30,000 MW was needed last hour, roughly 30,000 MW will be needed this hour. This autocorrelation is the strongest predictor. Temperature and time affect demand, but the baseline is set by recent actual demand.</p>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
      )}

      {/* Data Source Footer */}
      <footer className="data-footer">
        <div className="footer-content">
          <div className="footer-section">
            <h4>Data Source</h4>
            <p>Real UK electricity demand from <a href={DATA_INFO.url} target="_blank" rel="noopener noreferrer">National Grid ESO/NESO</a></p>
            <p className="footer-stats">{DATA_INFO.totalHours} hours of historical data ({DATA_INFO.years})</p>
          </div>
          <div className="footer-section">
            <h4>Features</h4>
            <p>Demand (MW) + Weather data (temperature, humidity, wind speed)</p>
            <p className="footer-stats">Weather from UK Met Office via Open-Meteo API</p>
          </div>
          <div className="footer-section">
            <h4>Models</h4>
            <p>XGBoost, Random Forest, Linear Regression, Explainable Boosting Machine</p>
            <p className="footer-stats">EBM provides full interpretability - see exactly how each feature affects predictions</p>
          </div>
        </div>
      </footer>

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

function formatFeatureName(feat) {
  const names = {
    'hour': 'Hour of Day',
    'dow': 'Day of Week',
    'month': 'Month',
    'lag_1': 'Previous Hour Demand',
    'lag_24': 'Yesterday Same Hour',
    'lag_168': 'Last Week Same Hour',
    'roll_24_mean': '24h Average Demand',
    'temp': 'Temperature',
    'temp_lag_24': 'Temp 24h Ago',
    'humidity': 'Humidity',
    'wind_speed': 'Wind Speed',
  }
  return names[feat] || feat.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

function getFeatureIcon(feat) {
  const icons = {
    'hour': '🕐',
    'dow': '📅',
    'month': '📆',
    'temp': '🌡️',
    'humidity': '💧',
    'wind_speed': '💨',
    'lag_1': '⚡',
    'lag_24': '📈',
    'lag_168': '📊',
    'roll_24_mean': '📉',
  }
  return icons[feat] || '📊'
}

function formatFeatureValue(feat, value) {
  if (value === undefined || value === null) return '-'
  if (feat === 'hour') return `${Math.round(value)}:00`
  if (feat === 'dow') {
    const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    return days[Math.round(value)] || value
  }
  if (feat === 'month') {
    const months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months[Math.round(value)] || value
  }
  if (feat === 'temp') return `${value.toFixed(1)}°C`
  if (feat === 'humidity') return `${value.toFixed(0)}%`
  if (feat === 'wind_speed') return `${value.toFixed(1)} km/h`
  if (feat.startsWith('lag_') || feat.startsWith('roll_')) return `${(value/1000).toFixed(1)}k MW`
  return value.toFixed(1)
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
