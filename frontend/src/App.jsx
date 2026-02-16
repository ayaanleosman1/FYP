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
  ReferenceLine,
  Cell
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

const SHAP_MODEL_NAMES = {
  xgb: 'XGBoost',
  rf: 'Random Forest',
  linear: 'Linear Regression',
  ebm: 'EBM',
  hybrid: 'Hybrid (XGB Residuals)'
}

const DATA_INFO = {
  source: 'UK National Grid (NESO)',
  years: '2009-2024',
  totalHours: '140,240',
  features: 'Demand + Weather (temperature, humidity, wind)',
  url: 'https://www.neso.energy/data-portal/historic-demand-data'
}

const WHATIF_HORIZONS = { H: 24, D: 7, W: 4, M: 3 }

const FEATURE_CATEGORIES = {
  weather: {
    label: 'Weather',
    icon: '🌤️',
    features: ['temp', 'humidity', 'wind_speed', 'solar_rad', 'direct_rad', 'temp_lag_24', 'solar_rad_lag_24', 'temp_lag_7', 'temp_roll_7']
  },
  calendar: {
    label: 'Calendar',
    icon: '📅',
    features: ['hour', 'dow', 'month', 'is_holiday', 'is_weekend', 'day_of_year', 'has_holiday', 'week_of_year', 'quarter']
  },
  lag: {
    label: 'Lag & Rolling',
    icon: '📈',
    features: ['lag_1', 'lag_7', 'lag_12', 'lag_24', 'lag_52', 'lag_168', 'roll_3_mean', 'roll_4_mean', 'roll_7_mean', 'roll_12_mean', 'roll_24_mean', 'roll_30_mean']
  },
  energy: {
    label: 'Energy Mix',
    icon: '⚡',
    features: ['gen_gas', 'gen_wind', 'gen_solar', 'gen_nuclear', 'carbon_intensity']
  }
}

const SCENARIO_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

function App() {
  const [granularities, setGranularities] = useState([])
  const [available, setAvailable] = useState({})
  const [selectedGranularity, setSelectedGranularity] = useState('D')
  const [allModelsData, setAllModelsData] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('dashboard')
  const [forecastView, setForecastView] = useState('overview')
  // SHAP state
  const [shapData, setShapData] = useState(null)
  const [shapLoading, setShapLoading] = useState(false)
  const [shapModel, setShapModel] = useState('xgb')
  const [shapAvailableModels, setShapAvailableModels] = useState([])
  // EBM shapes state
  const [ebmShapes, setEbmShapes] = useState(null)
  const [ebmLoading, setEbmLoading] = useState(false)
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
  const [whatIfGranularity, setWhatIfGranularity] = useState('D')
  const [whatIfHorizon, setWhatIfHorizon] = useState(7)
  const [sensitivityFeature, setSensitivityFeature] = useState(null)
  const [sensitivityData, setSensitivityData] = useState(null)
  const [sensitivityLoading, setSensitivityLoading] = useState(false)
  const [expandedCategories, setExpandedCategories] = useState({ weather: true, calendar: true, lag: false, energy: false })
  const [savedScenarios, setSavedScenarios] = useState([])

  // Dashboard state
  const [dashboardData, setDashboardData] = useState(null)
  const [dashboardLoading, setDashboardLoading] = useState(true)

  // Fetch dashboard data on mount
  useEffect(() => {
    fetch(`${API_BASE}/dashboard`)
      .then(r => r.json())
      .then(data => { setDashboardData(data); setDashboardLoading(false) })
      .catch(() => setDashboardLoading(false))
  }, [])

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
  const loadWhatIfFeatures = async (gran = whatIfGranularity) => {
    const horizon = WHATIF_HORIZONS[gran] || 24
    try {
      const res = await fetch(`${API_BASE}/whatif/features?granularity=${gran}&horizon=${horizon}`)
      if (res.ok) {
        const data = await res.json()
        setWhatIfFeatures(data)
        const initialValues = {}
        Object.entries(data.feature_ranges).forEach(([feat, range]) => {
          initialValues[feat] = range.median
        })
        setWhatIfValues(initialValues)
        fetchWhatIfPrediction(initialValues, true, gran)
      }
    } catch (err) {
      console.error('Failed to load what-if features:', err)
    }
  }

  // Fetch What-If prediction
  const fetchWhatIfPrediction = async (values, isBaseline = false, gran = whatIfGranularity) => {
    const horizon = WHATIF_HORIZONS[gran] || 24
    setWhatIfLoading(true)
    try {
      const res = await fetch(`${API_BASE}/whatif`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: values,
          granularity: gran,
          horizon: horizon
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

  // Fetch sensitivity analysis for a feature
  const fetchSensitivity = async (feature) => {
    setSensitivityFeature(feature)
    setSensitivityLoading(true)
    try {
      const res = await fetch(`${API_BASE}/whatif/sensitivity`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          feature,
          granularity: whatIfGranularity,
          horizon: WHATIF_HORIZONS[whatIfGranularity] || 24,
          steps: 30,
          base_features: whatIfValues
        })
      })
      if (res.ok) {
        const data = await res.json()
        setSensitivityData(data)
      }
    } catch (err) {
      console.error('Failed to fetch sensitivity:', err)
    } finally {
      setSensitivityLoading(false)
    }
  }

  // Save current scenario
  const saveScenario = () => {
    if (savedScenarios.length >= 4 || !whatIfPrediction) return
    const scenario = {
      id: Date.now(),
      name: `Scenario ${savedScenarios.length + 1}`,
      values: { ...whatIfValues },
      prediction: whatIfPrediction.prediction,
      contributions: { ...whatIfPrediction.contributions },
      timestamp: new Date().toLocaleTimeString()
    }
    setSavedScenarios(prev => [...prev, scenario])
  }

  // Remove a saved scenario
  const removeScenario = (id) => {
    setSavedScenarios(prev => prev.filter(s => s.id !== id))
  }

  // Get features available in a category for the current granularity
  const getAvailableFeaturesInCategory = (catKey) => {
    if (!whatIfFeatures) return []
    const catFeatures = FEATURE_CATEGORIES[catKey]?.features || []
    return catFeatures.filter(f => whatIfFeatures.features.includes(f))
  }

  // Handle slider change
  const handleWhatIfChange = (feature, value) => {
    const newValues = { ...whatIfValues, [feature]: parseFloat(value) }
    setWhatIfValues(newValues)
    fetchWhatIfPrediction(newValues)
  }

  // Load SHAP data for a specific model
  const loadShapData = async (modelOverride) => {
    setShapLoading(true)
    const m = modelOverride || shapModel
    try {
      const horizonMap = { H: 24, D: 7, W: 4, M: 3 }
      const horizon = horizonMap[selectedGranularity] || 7
      const res = await fetch(`${API_BASE}/shap?granularity=${selectedGranularity}&horizon=${horizon}&model=${m}`)
      if (res.ok) {
        const data = await res.json()
        setShapData(data)
      } else {
        setShapData(null)
      }
    } catch (err) {
      console.error('Failed to load SHAP data:', err)
      setShapData(null)
    } finally {
      setShapLoading(false)
    }
  }

  // Load which models have SHAP data available
  const loadShapAvailable = async () => {
    try {
      const horizonMap = { H: 24, D: 7, W: 4, M: 3 }
      const horizon = horizonMap[selectedGranularity] || 7
      const res = await fetch(`${API_BASE}/shap/available?granularity=${selectedGranularity}&horizon=${horizon}`)
      if (res.ok) {
        const data = await res.json()
        setShapAvailableModels(data.models || [])
      }
    } catch (err) {
      setShapAvailableModels([])
    }
  }

  // Load EBM shapes data
  const loadEbmShapes = async () => {
    setEbmLoading(true)
    try {
      const horizonMap = { H: 24, D: 7, W: 4, M: 3 }
      const horizon = horizonMap[selectedGranularity] || 7
      const res = await fetch(`${API_BASE}/ebm-shapes?granularity=${selectedGranularity}&horizon=${horizon}`)
      if (res.ok) {
        const data = await res.json()
        setEbmShapes(data)
      } else {
        setEbmShapes(null)
      }
    } catch (err) {
      console.error('Failed to load EBM shapes:', err)
      setEbmShapes(null)
    } finally {
      setEbmLoading(false)
    }
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
            className={`nav-link ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            Dashboard
          </button>
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
              if (!whatIfFeatures) loadWhatIfFeatures(whatIfGranularity)
            }}
          >
            What-If Analysis
          </button>
          <button
            className={`nav-link ${activeTab === 'shap' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('shap')
              loadShapData()
              loadShapAvailable()
            }}
          >
            SHAP Analysis
          </button>
          <button
            className={`nav-link ${activeTab === 'ebm' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('ebm')
              loadEbmShapes()
            }}
          >
            EBM Shapes
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

      {/* Dashboard Section */}
      {activeTab === 'dashboard' && (
        <div className="dashboard-section">
          {dashboardLoading ? (
            <div className="loading"><div className="spinner"></div>Loading dashboard...</div>
          ) : dashboardData ? (
            <>
              {/* Hero Stats */}
              <div className="hero-stats">
                <div className="hero-card">
                  <div className="hero-value">{dashboardData.stats.data_years}</div>
                  <div className="hero-label">Years of Data</div>
                  <div className="hero-sub">UK National Grid (2009-2024)</div>
                </div>
                <div className="hero-card">
                  <div className="hero-value">{dashboardData.stats.n_models}</div>
                  <div className="hero-label">ML Models</div>
                  <div className="hero-sub">Trained &amp; Compared</div>
                </div>
                <div className="hero-card hero-card-accent">
                  <div className="hero-value">{(100 - dashboardData.stats.best_smape).toFixed(1)}%</div>
                  <div className="hero-label">Best Accuracy</div>
                  <div className="hero-sub">
                    {MODEL_NAMES[dashboardData.stats.best_model]} &mdash; {({H:'Hourly',D:'Daily',W:'Weekly',M:'Monthly'})[dashboardData.stats.best_granularity]}
                  </div>
                </div>
                <div className="hero-card">
                  <div className="hero-value">{dashboardData.stats.n_features}</div>
                  <div className="hero-label">Features</div>
                  <div className="hero-sub">Weather, Lags, Calendar, Energy</div>
                </div>
              </div>

              {/* Performance Heatmap */}
              <div className="heatmap-section">
                <h2>Model Performance Across Time Scales</h2>
                <p className="heatmap-subtitle">SMAPE (%) &mdash; lower is better. Click any cell to explore.</p>
                <div className="heatmap-grid">
                  <div className="heatmap-header-row">
                    <div className="heatmap-corner"></div>
                    {['H', 'D', 'W', 'M'].map(g => (
                      <div key={g} className="heatmap-col-header">
                        {({H:'Hourly',D:'Daily',W:'Weekly',M:'Monthly'})[g]}
                      </div>
                    ))}
                  </div>
                  {['xgb', 'rf', 'linear', 'ebm'].map(model => {
                    return (
                      <div key={model} className="heatmap-row">
                        <div className="heatmap-row-label" style={{color: MODEL_COLORS[model]}}>
                          {MODEL_NAMES[model]}
                        </div>
                        {['H', 'D', 'W', 'M'].map(g => {
                          const cell = dashboardData.performance[g]?.[model]
                          const smape = cell?.smape
                          const granModels = dashboardData.performance[g] || {}
                          const bestInGran = Object.entries(granModels)
                            .reduce((best, [m, d]) => (!best || d.smape < best.smape) ? {model: m, smape: d.smape} : best, null)
                          const isBest = bestInGran?.model === model
                          const getHeatColor = (val) => {
                            if (val == null) return 'transparent'
                            const t = Math.min(Math.max((val - 2) / 12, 0), 1)
                            return `rgba(${Math.round(34 + t * 205)}, ${Math.round(197 - t * 128)}, ${Math.round(94 - t * 26)}, 0.25)`
                          }
                          return (
                            <div key={g} className={`heatmap-cell ${isBest ? 'heatmap-best' : ''}`}
                              style={{background: getHeatColor(smape)}}
                              onClick={() => { setSelectedGranularity(g); setActiveTab('forecast'); setForecastView('comparison') }}
                            >
                              {smape != null ? (
                                <>
                                  <span className="heatmap-value">{smape.toFixed(2)}%</span>
                                  {isBest && <span className="heatmap-crown">&#9733;</span>}
                                </>
                              ) : <span className="heatmap-na">N/A</span>}
                            </div>
                          )
                        })}
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Best Forecast Preview */}
              <div className="chart-section dashboard-chart">
                <div className="dashboard-chart-header">
                  <div>
                    <h3>Best Daily Forecast Preview</h3>
                    <p className="chart-subtitle">
                      {MODEL_NAMES[dashboardData.best_forecast.model]} &mdash; Daily (SMAPE: {dashboardData.best_forecast.smape?.toFixed(2)}%)
                    </p>
                  </div>
                  <button className="explore-link" onClick={() => {
                    setSelectedGranularity('D')
                    setSelectedModel(dashboardData.best_forecast.model)
                    setActiveTab('forecast')
                    setForecastView('overview')
                  }}>
                    Explore all models &#8594;
                  </button>
                </div>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={dashboardData.best_forecast.series?.map(s => ({
                    time: formatTime(s.t, 'D'), actual: s.actual, predicted: s.predicted,
                  }))}>
                    <defs>
                      <linearGradient id="dashActual" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="dashPred" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="time" stroke="#94a3b8" tick={{fill:'#94a3b8', fontSize:11}} />
                    <YAxis stroke="#94a3b8" tick={{fill:'#94a3b8'}} tickFormatter={formatNumber} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Area type="monotone" dataKey="actual" stroke="#6366f1" fill="url(#dashActual)" strokeWidth={2} name="Actual Demand" />
                    <Area type="monotone" dataKey="predicted" stroke="#10b981" fill="url(#dashPred)" strokeWidth={2} name="Predicted" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* What Drives Demand */}
              {dashboardData.top_features?.length > 0 && (
                <div className="features-section">
                  <div className="features-header">
                    <div>
                      <h3>What Drives UK Electricity Demand?</h3>
                      <p className="chart-subtitle">Top features by SHAP importance ({MODEL_NAMES[dashboardData.best_forecast.model]}, Daily)</p>
                    </div>
                    <button className="explore-link" onClick={() => {
                      setShapModel(dashboardData.best_forecast.model)
                      setActiveTab('shap')
                      loadShapData(dashboardData.best_forecast.model)
                      loadShapAvailable()
                    }}>
                      Deep dive into SHAP &#8594;
                    </button>
                  </div>
                  <div className="dashboard-features">
                    {dashboardData.top_features.map((feat) => {
                      const maxImp = dashboardData.top_features[0]?.importance || 1
                      const pct = (feat.importance / maxImp) * 100
                      const CAT_COLORS = { calendar: '#3b82f6', lag: '#8b5cf6', weather: '#f59e0b', energy: '#10b981', other: '#94a3b8' }
                      return (
                        <div key={feat.name} className="dashboard-feature-row">
                          <div className="dashboard-feature-label">
                            <span className="feature-dot" style={{background: CAT_COLORS[feat.category]}} />
                            {formatFeatureName(feat.name)}
                          </div>
                          <div className="dashboard-feature-bar-bg">
                            <div className="dashboard-feature-bar" style={{
                              width: `${pct}%`,
                              background: `linear-gradient(90deg, ${CAT_COLORS[feat.category]}, ${CAT_COLORS[feat.category]}88)`,
                            }} />
                            <span className="dashboard-feature-value">
                              {feat.importance.toLocaleString(undefined, {maximumFractionDigits: 0})}
                            </span>
                          </div>
                        </div>
                      )
                    })}
                    <div className="feature-legend">
                      {[{l:'Lag Features',c:'#8b5cf6'},{l:'Weather',c:'#f59e0b'},{l:'Calendar',c:'#3b82f6'},{l:'Energy Mix',c:'#10b981'}].map(item => (
                        <span key={item.l} className="feature-legend-item">
                          <span className="feature-dot" style={{background: item.c}} />{item.l}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Quick Access Cards */}
              <div className="quick-access">
                <h2>Explore Further</h2>
                <div className="quick-cards">
                  <div className="quick-card" onClick={() => { setActiveTab('whatif'); if (!whatIfFeatures) loadWhatIfFeatures(whatIfGranularity) }}>
                    <div className="quick-card-icon">&#127777;</div>
                    <div className="quick-card-title">What-If Analysis</div>
                    <div className="quick-card-desc">Adjust temperature, time of day, and see how demand changes in real-time</div>
                    <span className="quick-card-link">Explore &#8594;</span>
                  </div>
                  <div className="quick-card" onClick={() => { setActiveTab('shap'); loadShapData(); loadShapAvailable() }}>
                    <div className="quick-card-icon">&#128300;</div>
                    <div className="quick-card-title">SHAP Explanations</div>
                    <div className="quick-card-desc">Understand which features drive each prediction using game theory</div>
                    <span className="quick-card-link">Explore &#8594;</span>
                  </div>
                  <div className="quick-card" onClick={() => { setActiveTab('ebm'); loadEbmShapes() }}>
                    <div className="quick-card-icon">&#128270;</div>
                    <div className="quick-card-title">EBM Interpretability</div>
                    <div className="quick-card-desc">Glass-box model showing exact feature effects on demand predictions</div>
                    <span className="quick-card-link">Explore &#8594;</span>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="no-data"><h2>Could not load dashboard data</h2><p>Make sure the API server is running on port 8000.</p></div>
          )}
        </div>
      )}

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

              {/* Section 1: Granularity Selector */}
              <div className="whatif-gran-selector">
                {Object.entries(WHATIF_HORIZONS).map(([code, horizon]) => (
                  <button
                    key={code}
                    className={`whatif-gran-btn ${whatIfGranularity === code ? 'active' : ''}`}
                    onClick={() => {
                      setWhatIfGranularity(code)
                      setWhatIfHorizon(horizon)
                      setSensitivityFeature(null)
                      setSensitivityData(null)
                      setSavedScenarios([])
                      setWhatIfFeatures(null)
                      loadWhatIfFeatures(code)
                    }}
                  >
                    {({H:'Hourly',D:'Daily',W:'Weekly',M:'Monthly'})[code]}
                  </button>
                ))}
                <span className="whatif-horizon-label">
                  {whatIfHorizon} {({H:'hours',D:'days',W:'weeks',M:'months'})[whatIfGranularity]} ahead
                </span>
              </div>

              {!whatIfFeatures ? (
                <div className="loading"><div className="spinner"></div>Loading features...</div>
              ) : (
                <>
                  {/* Preset Scenarios */}
                  <div className="scenario-presets">
                    <h4>Quick Scenarios</h4>
                    <div className="preset-grid">
                      <button
                        className="preset-card"
                        onClick={() => {
                          const vals = {
                            hour: 8, dow: 1, month: 1, temp: 2, humidity: 85, wind_speed: 15,
                            lag_1: 38000, lag_24: 35000, lag_168: 37000, roll_24_mean: 34000,
                            gen_gas: 15000, gen_wind: 8000, gen_solar: 500, gen_nuclear: 5000, carbon_intensity: 250
                          }
                          const filtered = {}
                          Object.entries(vals).forEach(([k, v]) => { if (whatIfFeatures.features.includes(k)) filtered[k] = v })
                          Object.entries(whatIfFeatures.feature_ranges).forEach(([feat, range]) => { if (!(feat in filtered)) filtered[feat] = range.median })
                          setWhatIfValues(filtered)
                          fetchWhatIfPrediction(filtered)
                        }}
                      >
                        <span className="preset-icon">🌅</span>
                        <span className="preset-name">Cold Winter Morning</span>
                        <span className="preset-desc">8 AM, Monday, 2°C</span>
                      </button>
                      <button
                        className="preset-card"
                        onClick={() => {
                          const vals = {
                            hour: 18, dow: 2, month: 7, temp: 24, humidity: 55, wind_speed: 10,
                            lag_1: 30000, lag_24: 28000, lag_168: 29000, roll_24_mean: 27000,
                            gen_gas: 10000, gen_wind: 12000, gen_solar: 6000, gen_nuclear: 5000, carbon_intensity: 180
                          }
                          const filtered = {}
                          Object.entries(vals).forEach(([k, v]) => { if (whatIfFeatures.features.includes(k)) filtered[k] = v })
                          Object.entries(whatIfFeatures.feature_ranges).forEach(([feat, range]) => { if (!(feat in filtered)) filtered[feat] = range.median })
                          setWhatIfValues(filtered)
                          fetchWhatIfPrediction(filtered)
                        }}
                      >
                        <span className="preset-icon">🌇</span>
                        <span className="preset-name">Summer Evening Peak</span>
                        <span className="preset-desc">6 PM, Weekday, 24°C</span>
                      </button>
                      <button
                        className="preset-card"
                        onClick={() => {
                          const vals = {
                            hour: 3, dow: 6, month: 5, temp: 12, humidity: 75, wind_speed: 8,
                            lag_1: 18000, lag_24: 20000, lag_168: 18500, roll_24_mean: 22000,
                            gen_gas: 8000, gen_wind: 10000, gen_solar: 0, gen_nuclear: 5000, carbon_intensity: 160
                          }
                          const filtered = {}
                          Object.entries(vals).forEach(([k, v]) => { if (whatIfFeatures.features.includes(k)) filtered[k] = v })
                          Object.entries(whatIfFeatures.feature_ranges).forEach(([feat, range]) => { if (!(feat in filtered)) filtered[feat] = range.median })
                          setWhatIfValues(filtered)
                          fetchWhatIfPrediction(filtered)
                        }}
                      >
                        <span className="preset-icon">🌙</span>
                        <span className="preset-name">Weekend Night</span>
                        <span className="preset-desc">3 AM, Saturday, 12°C</span>
                      </button>
                      <button
                        className="preset-card"
                        onClick={() => {
                          const vals = {
                            hour: 17, dow: 3, month: 12, temp: -2, humidity: 90, wind_speed: 20,
                            lag_1: 42000, lag_24: 40000, lag_168: 41000, roll_24_mean: 38000,
                            gen_gas: 18000, gen_wind: 5000, gen_solar: 0, gen_nuclear: 5500, carbon_intensity: 300
                          }
                          const filtered = {}
                          Object.entries(vals).forEach(([k, v]) => { if (whatIfFeatures.features.includes(k)) filtered[k] = v })
                          Object.entries(whatIfFeatures.feature_ranges).forEach(([feat, range]) => { if (!(feat in filtered)) filtered[feat] = range.median })
                          setWhatIfValues(filtered)
                          fetchWhatIfPrediction(filtered)
                        }}
                      >
                        <span className="preset-icon">❄️</span>
                        <span className="preset-name">Peak Winter Evening</span>
                        <span className="preset-desc">5 PM, December, -2°C</span>
                      </button>
                    </div>
                  </div>

                  {/* Section 2: Result Panel + Category Sliders */}
                  <div className="whatif-container">
                    {/* Result Panel */}
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
                          {/* Visual Gauge - dynamic range from lag_1 or feature ranges */}
                          {(() => {
                            const lagRange = whatIfFeatures.feature_ranges['lag_1'] || whatIfFeatures.feature_ranges[whatIfFeatures.features[0]]
                            const gaugeMin = lagRange ? Math.floor(lagRange.min / 1000) * 1000 : 15000
                            const gaugeMax = lagRange ? Math.ceil(lagRange.max / 1000) * 1000 : 45000
                            const gaugeRange = gaugeMax - gaugeMin
                            const gaugeRatio = Math.max(0, Math.min(1, (whatIfPrediction.prediction - gaugeMin) / gaugeRange))
                            return (
                              <div className="demand-gauge">
                                <svg viewBox="0 0 200 120" className="gauge-svg">
                                  <defs>
                                    <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                      <stop offset="0%" stopColor="#10b981" />
                                      <stop offset="50%" stopColor="#f59e0b" />
                                      <stop offset="100%" stopColor="#ef4444" />
                                    </linearGradient>
                                  </defs>
                                  <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#334155" strokeWidth="12" strokeLinecap="round" />
                                  <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#gaugeGradient)" strokeWidth="12" strokeLinecap="round"
                                    strokeDasharray={`${gaugeRatio * 251} 251`} />
                                  <line x1="100" y1="100"
                                    x2={100 + 60 * Math.cos(Math.PI - gaugeRatio * Math.PI)}
                                    y2={100 - 60 * Math.sin(Math.PI - gaugeRatio * Math.PI)}
                                    stroke="#f1f5f9" strokeWidth="3" strokeLinecap="round" />
                                  <circle cx="100" cy="100" r="8" fill="#f1f5f9" />
                                </svg>
                                <div className="gauge-labels">
                                  <span>{formatNumber(gaugeMin)}</span>
                                  <span>{formatNumber((gaugeMin + gaugeMax) / 2)}</span>
                                  <span>{formatNumber(gaugeMax)}</span>
                                </div>
                              </div>
                            )
                          })()}

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

                          {/* Save Scenario Button */}
                          <button
                            className="save-scenario-btn"
                            onClick={saveScenario}
                            disabled={savedScenarios.length >= 4}
                          >
                            {savedScenarios.length >= 4 ? 'Max 4 Scenarios' : 'Save Scenario'}
                          </button>
                        </>
                      ) : null}
                    </div>

                    {/* Sliders Panel - Categorized */}
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

                      {Object.entries(FEATURE_CATEGORIES).map(([catKey, cat]) => {
                        const catFeats = getAvailableFeaturesInCategory(catKey)
                        if (catFeats.length === 0) return null
                        const isExpanded = expandedCategories[catKey]
                        return (
                          <div key={catKey} className="feature-category">
                            <button
                              className="category-header"
                              onClick={() => setExpandedCategories(prev => ({ ...prev, [catKey]: !prev[catKey] }))}
                            >
                              <span className="category-icon">{cat.icon}</span>
                              <span className="category-label">{cat.label}</span>
                              <span className="category-count">{catFeats.length}</span>
                              <span className={`category-chevron ${isExpanded ? 'expanded' : ''}`}>&#9656;</span>
                            </button>
                            {isExpanded && (
                              <div className="category-sliders">
                                {catFeats.map(feat => {
                                  const range = whatIfFeatures.feature_ranges[feat]
                                  if (!range) return null
                                  const importance = whatIfFeatures.feature_importances[feat] || 0
                                  const maxImportance = Math.max(...Object.values(whatIfFeatures.feature_importances || { _: 1 }))
                                  return (
                                    <div key={feat} className="whatif-slider">
                                      <div className="slider-header">
                                        <label>
                                          <span className="feature-icon">{getFeatureIcon(feat)}</span>
                                          {formatFeatureName(feat)}
                                        </label>
                                        <div className="slider-header-right">
                                          <span className="slider-value">{formatFeatureValue(feat, whatIfValues[feat])}</span>
                                          <button
                                            className="sensitivity-btn"
                                            title="Sensitivity analysis"
                                            onClick={() => fetchSensitivity(feat)}
                                          >
                                            &#128200;
                                          </button>
                                        </div>
                                      </div>
                                      <div className="slider-track-container">
                                        <input
                                          type="range"
                                          min={range.min}
                                          max={range.max}
                                          step={['hour', 'dow', 'month', 'is_holiday', 'is_weekend', 'day_of_year', 'quarter', 'week_of_year'].includes(feat) ? 1 : (range.max - range.min) / 100}
                                          value={whatIfValues[feat] ?? range.median}
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
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>

                  {/* Section 3: Contribution Waterfall Chart */}
                  {whatIfPrediction && Object.keys(whatIfPrediction.contributions || {}).length > 0 && (
                    <div className="waterfall-section">
                      <h4>Feature Contribution Waterfall</h4>
                      <p className="waterfall-subtitle">How each feature pushes the prediction up (green) or down (red)</p>
                      <ResponsiveContainer width="100%" height={Math.max(300, Object.keys(whatIfPrediction.contributions).length * 28)}>
                        <BarChart
                          data={Object.entries(whatIfPrediction.contributions)
                            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                            .map(([feat, val]) => ({ name: formatFeatureName(feat), value: Math.round(val), raw: val }))}
                          layout="vertical"
                          margin={{ top: 5, right: 40, left: 140, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                          <XAxis type="number" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={formatNumber} />
                          <YAxis type="category" dataKey="name" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} width={135} />
                          <Tooltip
                            content={({ active, payload }) => {
                              if (!active || !payload?.[0]) return null
                              const d = payload[0].payload
                              return (
                                <div className="custom-tooltip">
                                  <div className="tooltip-label">{d.name}</div>
                                  <div className="tooltip-row" style={{ color: d.raw >= 0 ? '#10b981' : '#ef4444' }}>
                                    <span className="tooltip-value">{d.raw >= 0 ? '+' : ''}{formatNumber(d.raw)} MW</span>
                                  </div>
                                </div>
                              )
                            }}
                          />
                          <ReferenceLine x={0} stroke="#64748b" />
                          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                            {Object.entries(whatIfPrediction.contributions)
                              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                              .map(([feat, val], idx) => (
                                <Cell key={feat} fill={val >= 0 ? '#10b981' : '#ef4444'} fillOpacity={0.8} />
                              ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* Section 4: Sensitivity Analysis */}
                  {sensitivityFeature && (
                    <div className="sensitivity-section">
                      <div className="sensitivity-header">
                        <h4>Sensitivity: {formatFeatureName(sensitivityFeature)}</h4>
                        <button className="sensitivity-close" onClick={() => { setSensitivityFeature(null); setSensitivityData(null) }}>&#10005;</button>
                      </div>
                      {sensitivityLoading ? (
                        <div className="loading"><div className="spinner"></div>Calculating sensitivity...</div>
                      ) : sensitivityData ? (
                        <ResponsiveContainer width="100%" height={300}>
                          <ComposedChart data={sensitivityData.sweep} margin={{ top: 10, right: 30, left: 20, bottom: 10 }}>
                            <defs>
                              <linearGradient id="sensitivityGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.4} />
                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.05} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis
                              dataKey="value"
                              stroke="#94a3b8"
                              tick={{ fill: '#94a3b8', fontSize: 11 }}
                              tickFormatter={(v) => formatShapeXAxis(sensitivityFeature, v)}
                              label={{ value: formatFeatureName(sensitivityFeature), position: 'bottom', fill: '#94a3b8', fontSize: 12 }}
                            />
                            <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} tickFormatter={formatNumber} />
                            <Tooltip
                              content={({ active, payload }) => {
                                if (!active || !payload?.[0]) return null
                                const d = payload[0].payload
                                return (
                                  <div className="custom-tooltip">
                                    <div className="tooltip-label">{formatFeatureName(sensitivityFeature)}: {formatFeatureValue(sensitivityFeature, d.value)}</div>
                                    <div className="tooltip-row" style={{ color: '#8b5cf6' }}>
                                      <span className="tooltip-name">Prediction:</span>
                                      <span className="tooltip-value">{formatNumber(d.prediction)} MW</span>
                                    </div>
                                  </div>
                                )
                              }}
                            />
                            <Area type="monotone" dataKey="prediction" stroke="#8b5cf6" strokeWidth={2} fill="url(#sensitivityGrad)" />
                            {whatIfValues[sensitivityFeature] !== undefined && (
                              <ReferenceLine
                                x={whatIfValues[sensitivityFeature]}
                                stroke="#f59e0b"
                                strokeDasharray="5 5"
                                strokeWidth={2}
                                label={{ value: 'Current', fill: '#f59e0b', fontSize: 11, position: 'top' }}
                              />
                            )}
                          </ComposedChart>
                        </ResponsiveContainer>
                      ) : null}
                    </div>
                  )}

                  {/* Section 5: Scenario Comparison */}
                  {savedScenarios.length > 0 && (
                    <div className="scenario-comparison-section">
                      <h4>Scenario Comparison</h4>
                      <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={savedScenarios.map(s => ({ name: s.name, prediction: s.prediction }))} margin={{ top: 10, right: 30, left: 20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                          <XAxis dataKey="name" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                          <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} tickFormatter={formatNumber} />
                          <Tooltip
                            content={({ active, payload }) => {
                              if (!active || !payload?.[0]) return null
                              return (
                                <div className="custom-tooltip">
                                  <div className="tooltip-label">{payload[0].payload.name}</div>
                                  <div className="tooltip-row" style={{ color: '#8b5cf6' }}>
                                    <span className="tooltip-value">{formatNumber(payload[0].value)} MW</span>
                                  </div>
                                </div>
                              )
                            }}
                          />
                          <Bar dataKey="prediction" radius={[6, 6, 0, 0]}>
                            {savedScenarios.map((s, idx) => (
                              <Cell key={s.id} fill={SCENARIO_COLORS[idx % SCENARIO_COLORS.length]} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>

                      <div className="scenario-cards">
                        {savedScenarios.map((scenario, idx) => (
                          <div key={scenario.id} className="scenario-card" style={{ borderLeftColor: SCENARIO_COLORS[idx % SCENARIO_COLORS.length] }}>
                            <div className="scenario-card-header">
                              <input
                                className="scenario-name-input"
                                value={scenario.name}
                                onChange={(e) => {
                                  setSavedScenarios(prev => prev.map(s => s.id === scenario.id ? { ...s, name: e.target.value } : s))
                                }}
                              />
                              <span className="scenario-time">{scenario.timestamp}</span>
                            </div>
                            <div className="scenario-prediction">
                              {Math.round(scenario.prediction).toLocaleString()} <span>MW</span>
                            </div>
                            <div className="scenario-actions">
                              <button
                                className="scenario-load-btn"
                                onClick={() => {
                                  setWhatIfValues(scenario.values)
                                  fetchWhatIfPrediction(scenario.values)
                                }}
                              >
                                Load
                              </button>
                              <button
                                className="scenario-remove-btn"
                                onClick={() => removeScenario(scenario.id)}
                              >
                                &#10005;
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
      )}

      {/* SHAP Analysis Section */}
      {activeTab === 'shap' && (
        <div className="shap-section">
          <div className="shap-header">
            <h2>SHAP Analysis - {SHAP_MODEL_NAMES[shapModel] || shapModel}</h2>
            <p className="shap-description">
              SHAP (SHapley Additive exPlanations) shows how each feature contributes to predictions.
              Based on game theory, it fairly distributes the prediction among features.
            </p>
            <div className="shap-model-selector">
              {['xgb', 'rf', 'linear', 'ebm', 'hybrid'].map(m => (
                <button
                  key={m}
                  className={`${shapModel === m ? 'active' : ''} ${!shapAvailableModels.includes(m) ? 'disabled' : ''}`}
                  disabled={!shapAvailableModels.includes(m)}
                  onClick={() => {
                    setShapModel(m)
                    loadShapData(m)
                  }}
                >
                  {SHAP_MODEL_NAMES[m]}
                </button>
              ))}
            </div>
            <div className="shap-granularity-selector">
              {['H', 'D', 'W', 'M'].map(g => (
                <button
                  key={g}
                  className={selectedGranularity === g ? 'active' : ''}
                  onClick={() => {
                    setSelectedGranularity(g)
                    setTimeout(() => {
                      loadShapData()
                      loadShapAvailable()
                    }, 100)
                  }}
                >
                  {{H: 'Hourly', D: 'Daily', W: 'Weekly', M: 'Monthly'}[g]}
                </button>
              ))}
            </div>
          </div>

          {shapLoading ? (
            <div className="loading">
              <div className="spinner"></div>
              Loading SHAP analysis...
            </div>
          ) : shapData ? (
            <div className="shap-content">
              {/* Feature Importance Bar Chart */}
              <div className="shap-chart-section">
                <h3>Global Feature Importance</h3>
                <p className="chart-subtitle">Mean absolute SHAP value (higher = more impact on predictions)</p>
                <div className="shap-bars">
                  {shapData.distribution?.map((feat, idx) => {
                    const maxImp = shapData.distribution[0]?.importance || 1
                    const pct = (feat.importance / maxImp) * 100
                    return (
                      <div key={feat.feature} className="shap-bar-row">
                        <div className="shap-bar-label">{formatFeatureName(feat.feature)}</div>
                        <div className="shap-bar-container">
                          <div
                            className="shap-bar-fill"
                            style={{
                              width: `${pct}%`,
                              backgroundColor: idx === 0 ? '#ef4444' : idx < 3 ? '#f59e0b' : '#3b82f6'
                            }}
                          />
                          <span className="shap-bar-value">{feat.importance.toLocaleString()}</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Beeswarm-style scatter plot for top 3 features */}
              <div className="shap-chart-section">
                <h3>Feature Impact Distribution</h3>
                <p className="chart-subtitle">Each dot is a prediction. X-axis shows SHAP impact, color shows feature value.</p>
                <div className="shap-distributions">
                  {shapData.distribution?.slice(0, 5).map(feat => (
                    <div key={feat.feature} className="shap-dist-row">
                      <div className="shap-dist-label">{formatFeatureName(feat.feature)}</div>
                      <div className="shap-dist-plot">
                        <svg viewBox="-100 0 200 30" className="shap-scatter">
                          {feat.shap_values?.map((sv, i) => {
                            const fv = feat.values[i]
                            const normalizedFv = (fv - feat.min_val) / (feat.max_val - feat.min_val || 1)
                            const maxShap = Math.max(...feat.shap_values.map(Math.abs)) || 1
                            const x = (sv / maxShap) * 90
                            const color = `hsl(${240 - normalizedFv * 240}, 80%, 50%)`
                            return (
                              <circle
                                key={i}
                                cx={x}
                                cy={15 + (Math.random() - 0.5) * 20}
                                r="3"
                                fill={color}
                                opacity="0.6"
                              />
                            )
                          })}
                          <line x1="0" y1="0" x2="0" y2="30" stroke="#64748b" strokeWidth="1" />
                        </svg>
                        <div className="shap-dist-legend">
                          <span className="low">Low</span>
                          <div className="gradient-bar"></div>
                          <span className="high">High</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Insights */}
              <div className="shap-insights">
                <h3>Key Insights</h3>
                <div className="insight-cards">
                  <div className="insight-card">
                    <div className="insight-icon">📊</div>
                    <div className="insight-text">
                      <strong>{formatFeatureName(shapData.features?.[0])}</strong> is the most important feature,
                      contributing {((shapData.importance?.[0] / shapData.importance?.reduce((a,b) => a+b, 0)) * 100).toFixed(1)}% of total importance.
                    </div>
                  </div>
                  <div className="insight-card">
                    <div className="insight-icon">🔬</div>
                    <div className="insight-text">
                      Analysis based on <strong>{shapData.n_samples?.toLocaleString()}</strong> samples
                      from the {shapData.granularity_name} test set
                      using <strong>{shapData.model_name || SHAP_MODEL_NAMES[shapModel]}</strong>.
                    </div>
                  </div>
                  <div className="insight-card">
                    <div className="insight-icon">⚡</div>
                    <div className="insight-text">
                      Lag features dominate because electricity demand is highly autocorrelated -
                      current demand strongly predicts near-future demand.
                    </div>
                  </div>
                  {shapData.note && (
                    <div className="insight-card">
                      <div className="insight-icon">ℹ️</div>
                      <div className="insight-text">{shapData.note}</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="no-data">
              <h3>No SHAP data available for {SHAP_MODEL_NAMES[shapModel] || shapModel} at {selectedGranularity}</h3>
              <p>Run: <code>python ml/generate_shap.py -g {selectedGranularity} -m {shapModel}</code></p>
            </div>
          )}
        </div>
      )}

      {/* EBM Interpretability Section */}
      {activeTab === 'ebm' && (
        <div className="ebm-section">
          <div className="ebm-header">
            <h2>Model Interpretability</h2>
            <p className="ebm-description">
              See exactly how the EBM (Explainable Boosting Machine) makes predictions.
              Unlike black-box models, every prediction can be fully explained.
            </p>
            <div className="ebm-granularity-selector">
              {['H', 'D', 'W', 'M'].map(g => (
                <button
                  key={g}
                  className={selectedGranularity === g ? 'active' : ''}
                  onClick={() => {
                    setSelectedGranularity(g)
                    setTimeout(loadEbmShapes, 100)
                  }}
                >
                  {{H: 'Hourly', D: 'Daily', W: 'Weekly', M: 'Monthly'}[g]}
                </button>
              ))}
            </div>
          </div>

          {ebmLoading ? (
            <div className="loading">
              <div className="spinner"></div>
              Loading interpretability data...
            </div>
          ) : ebmShapes ? (
            <div className="ebm-content">
              {/* Key Discoveries - Plain English */}
              <div className="discoveries-section">
                <h3>What The Model Learned</h3>
                <div className="discovery-cards">
                  {generateInsights(ebmShapes).map((insight, idx) => (
                    <div key={idx} className="discovery-card">
                      <div className="discovery-icon">{insight.icon}</div>
                      <div className="discovery-content">
                        <div className="discovery-title">{insight.title}</div>
                        <div className="discovery-text">{insight.text}</div>
                      </div>
                      <div className={`discovery-impact ${insight.impact > 0 ? 'positive' : 'negative'}`}>
                        {insight.impact > 0 ? '+' : ''}{formatNumber(insight.impact)} MW
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Feature Ranking */}
              <div className="feature-ranking">
                <h3>What Matters Most?</h3>
                <p className="ranking-subtitle">Features ranked by their influence on predictions</p>
                <div className="ranking-list">
                  {ebmShapes.features?.slice(0, 6).map((feat, idx) => {
                    const maxImp = ebmShapes.features[0]?.importance || 1
                    const pct = (feat.importance / maxImp) * 100
                    return (
                      <div key={feat.name} className="ranking-item">
                        <div className="ranking-position">#{idx + 1}</div>
                        <div className="ranking-info">
                          <div className="ranking-name">{formatFeatureName(feat.name)}</div>
                          <div className="ranking-bar">
                            <div className="ranking-fill" style={{ width: `${pct}%` }} />
                          </div>
                        </div>
                        <div className="ranking-range">
                          <span className="range-down">{formatNumber(feat.min_effect)}</span>
                          <span className="range-sep">to</span>
                          <span className="range-up">+{formatNumber(feat.max_effect)}</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Example Prediction Breakdown */}
              <div className="prediction-breakdown">
                <h3>How A Prediction Is Made</h3>
                <p className="breakdown-subtitle">Example: Predicting demand for a typical winter weekday morning</p>
                <div className="breakdown-visual">
                  <div className="breakdown-base">
                    <span className="breakdown-label">Base Demand</span>
                    <span className="breakdown-value">30,000 MW</span>
                  </div>
                  <div className="breakdown-arrow">+</div>
                  <div className="breakdown-factors">
                    {[
                      { name: 'Cold temperature (5°C)', effect: 2100, icon: '🌡️' },
                      { name: 'Morning peak (8 AM)', effect: 1800, icon: '🌅' },
                      { name: 'Weekday', effect: 1200, icon: '📅' },
                      { name: 'January', effect: 800, icon: '❄️' },
                      { name: 'Low solar generation', effect: 400, icon: '☁️' },
                    ].map((factor, idx) => (
                      <div key={idx} className="breakdown-factor">
                        <span className="factor-icon">{factor.icon}</span>
                        <span className="factor-name">{factor.name}</span>
                        <span className={`factor-effect ${factor.effect >= 0 ? 'positive' : 'negative'}`}>
                          {factor.effect >= 0 ? '+' : ''}{factor.effect.toLocaleString()}
                        </span>
                      </div>
                    ))}
                  </div>
                  <div className="breakdown-arrow">=</div>
                  <div className="breakdown-result">
                    <span className="breakdown-label">Final Prediction</span>
                    <span className="breakdown-value">36,300 MW</span>
                  </div>
                </div>
              </div>

              {/* Why This Matters */}
              <div className="why-matters">
                <h3>Why Interpretability Matters</h3>
                <div className="matters-grid">
                  <div className="matters-card">
                    <div className="matters-icon">🔍</div>
                    <div className="matters-title">Audit & Trust</div>
                    <div className="matters-text">Grid operators can verify predictions make physical sense before acting on them</div>
                  </div>
                  <div className="matters-card">
                    <div className="matters-icon">⚠️</div>
                    <div className="matters-title">Error Detection</div>
                    <div className="matters-text">Unusual contributions reveal when the model might be wrong</div>
                  </div>
                  <div className="matters-card">
                    <div className="matters-icon">📊</div>
                    <div className="matters-title">Domain Insights</div>
                    <div className="matters-text">Learned patterns validate or challenge expert assumptions about demand drivers</div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="no-data">
              <h3>No interpretability data available for {selectedGranularity}</h3>
              <p>Run: <code>python ml/generate_ebm_shapes.py -g {selectedGranularity}</code></p>
            </div>
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
    'day_of_year': 'Day of Year',
    'is_holiday': 'Holiday',
    'is_weekend': 'Weekend',
    'has_holiday': 'Has Holiday',
    'week_of_year': 'Week of Year',
    'quarter': 'Quarter',
    'lag_1': 'Previous Period Demand',
    'lag_7': '7-Period Lag',
    'lag_12': '12-Period Lag',
    'lag_24': '24-Period Lag',
    'lag_52': '52-Period Lag',
    'lag_168': 'Week-Ago Demand',
    'roll_3_mean': '3-Period Rolling Avg',
    'roll_4_mean': '4-Period Rolling Avg',
    'roll_7_mean': '7-Period Rolling Avg',
    'roll_12_mean': '12-Period Rolling Avg',
    'roll_24_mean': '24-Period Rolling Avg',
    'roll_30_mean': '30-Period Rolling Avg',
    'temp': 'Temperature',
    'temp_lag_24': 'Temp 24h Ago',
    'temp_lag_7': 'Temp 7d Ago',
    'temp_roll_7': 'Temp 7d Avg',
    'humidity': 'Humidity',
    'wind_speed': 'Wind Speed',
    'solar_rad': 'Solar Radiation',
    'solar_rad_lag_24': 'Solar Rad 24h Ago',
    'direct_rad': 'Direct Radiation',
    'gen_gas': 'Gas Generation',
    'gen_wind': 'Wind Generation',
    'gen_solar': 'Solar Generation',
    'gen_nuclear': 'Nuclear Generation',
    'carbon_intensity': 'Carbon Intensity',
  }
  return names[feat] || feat.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

function getFeatureIcon(feat) {
  const icons = {
    'hour': '🕐', 'dow': '📅', 'month': '📆', 'day_of_year': '📅',
    'is_holiday': '🎄', 'is_weekend': '🛋️', 'has_holiday': '🎄',
    'week_of_year': '📅', 'quarter': '📅',
    'temp': '🌡️', 'humidity': '💧', 'wind_speed': '💨',
    'solar_rad': '☀️', 'direct_rad': '☀️', 'temp_lag_24': '🌡️',
    'temp_lag_7': '🌡️', 'temp_roll_7': '🌡️', 'solar_rad_lag_24': '☀️',
    'lag_1': '⚡', 'lag_7': '📈', 'lag_12': '📈', 'lag_24': '📈',
    'lag_52': '📈', 'lag_168': '📊',
    'roll_3_mean': '📉', 'roll_4_mean': '📉', 'roll_7_mean': '📉',
    'roll_12_mean': '📉', 'roll_24_mean': '📉', 'roll_30_mean': '📉',
    'gen_gas': '🔥', 'gen_wind': '💨', 'gen_solar': '☀️',
    'gen_nuclear': '⚛️', 'carbon_intensity': '🌍',
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
  if (feat === 'is_holiday' || feat === 'is_weekend' || feat === 'has_holiday') return Math.round(value) === 1 ? 'Yes' : 'No'
  if (feat === 'quarter') return `Q${Math.round(value)}`
  if (feat === 'day_of_year' || feat === 'week_of_year') return Math.round(value).toString()
  if (feat === 'temp' || feat.startsWith('temp_')) return `${value.toFixed(1)}°C`
  if (feat === 'humidity') return `${value.toFixed(0)}%`
  if (feat === 'wind_speed') return `${value.toFixed(1)} km/h`
  if (feat === 'solar_rad' || feat === 'direct_rad' || feat.startsWith('solar_rad_')) return `${value.toFixed(0)} W/m²`
  if (feat === 'carbon_intensity') return `${value.toFixed(0)} g/kWh`
  if (feat.startsWith('gen_')) return `${(value/1000).toFixed(1)}k MW`
  if (feat.startsWith('lag_') || feat.startsWith('roll_')) return `${(value/1000).toFixed(1)}k MW`
  return value.toFixed(1)
}

function formatShapeXAxis(feat, value) {
  if (value === undefined || value === null) return ''
  if (feat === 'hour') return `${Math.round(value)}h`
  if (feat === 'dow') {
    const days = ['S', 'M', 'T', 'W', 'T', 'F', 'S']
    return days[Math.round(value)] || value
  }
  if (feat === 'month') return Math.round(value)
  if (feat === 'temp' || feat === 'temp_lag_24') return `${Math.round(value)}°`
  if (feat === 'humidity') return `${Math.round(value)}%`
  if (feat.startsWith('lag_') || feat.startsWith('roll_') || feat.startsWith('gen_')) {
    return `${(value/1000).toFixed(0)}k`
  }
  if (typeof value === 'number') return Math.abs(value) >= 1000 ? `${(value/1000).toFixed(0)}k` : Math.round(value)
  return value
}

function generateInsights(ebmShapes) {
  const insights = []
  const features = ebmShapes.features || []

  // Find hour feature
  const hourFeat = features.find(f => f.name === 'hour')
  if (hourFeat) {
    const maxIdx = hourFeat.y.indexOf(Math.max(...hourFeat.y))
    const minIdx = hourFeat.y.indexOf(Math.min(...hourFeat.y))
    const peakHour = Math.round(hourFeat.x[maxIdx])
    const lowHour = Math.round(hourFeat.x[minIdx])
    insights.push({
      icon: '⏰',
      title: 'Peak Hours',
      text: `Demand peaks around ${peakHour}:00 and is lowest at ${lowHour}:00`,
      impact: hourFeat.max_effect - hourFeat.min_effect
    })
  }

  // Find temperature feature
  const tempFeat = features.find(f => f.name === 'temp')
  if (tempFeat) {
    const coldEffect = tempFeat.y[0] || 0  // Effect at low temp
    const warmEffect = tempFeat.y[tempFeat.y.length - 1] || 0  // Effect at high temp
    insights.push({
      icon: '🌡️',
      title: 'Temperature Effect',
      text: `Cold weather increases demand, warm weather decreases it`,
      impact: coldEffect - warmEffect
    })
  }

  // Find day of week feature
  const dowFeat = features.find(f => f.name === 'dow')
  if (dowFeat) {
    const weekdayAvg = (dowFeat.y[1] + dowFeat.y[2] + dowFeat.y[3] + dowFeat.y[4] + dowFeat.y[5]) / 5
    const weekendAvg = (dowFeat.y[0] + dowFeat.y[6]) / 2
    insights.push({
      icon: '📅',
      title: 'Weekday vs Weekend',
      text: `Weekdays have ${weekdayAvg > weekendAvg ? 'higher' : 'lower'} demand than weekends`,
      impact: weekdayAvg - weekendAvg
    })
  }

  // Find holiday feature
  const holidayFeat = features.find(f => f.name === 'is_holiday')
  if (holidayFeat && holidayFeat.y.length >= 2) {
    insights.push({
      icon: '🎄',
      title: 'Holiday Effect',
      text: `Holidays significantly reduce electricity demand`,
      impact: holidayFeat.y[1] - holidayFeat.y[0]  // Holiday effect vs non-holiday
    })
  }

  // Find solar feature
  const solarFeat = features.find(f => f.name === 'gen_solar')
  if (solarFeat) {
    insights.push({
      icon: '☀️',
      title: 'Solar Generation',
      text: `High solar generation reduces grid demand`,
      impact: solarFeat.min_effect - solarFeat.max_effect
    })
  }

  // Find month feature
  const monthFeat = features.find(f => f.name === 'month')
  if (monthFeat) {
    const winterAvg = ((monthFeat.y[0] || 0) + (monthFeat.y[1] || 0) + (monthFeat.y[11] || 0)) / 3
    const summerAvg = ((monthFeat.y[5] || 0) + (monthFeat.y[6] || 0) + (monthFeat.y[7] || 0)) / 3
    insights.push({
      icon: '❄️',
      title: 'Seasonal Pattern',
      text: `Winter months have higher demand than summer`,
      impact: winterAvg - summerAvg
    })
  }

  return insights.slice(0, 5)  // Top 5 insights
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
