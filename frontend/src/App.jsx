import { useState, useEffect } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

const GRANULARITY_LABELS = {
  H: 'Hourly',
  D: 'Daily',
  W: 'Weekly',
  M: 'Monthly',
  Y: 'Yearly'
}

function App() {
  const [granularities, setGranularities] = useState([])
  const [available, setAvailable] = useState({})
  const [selectedGranularity, setSelectedGranularity] = useState('H')
  const [selectedModel, setSelectedModel] = useState('xgb')
  const [selectedHorizon, setSelectedHorizon] = useState(24)
  const [metrics, setMetrics] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

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
      .catch(err => setError('Failed to connect to API. Is the server running?'))
  }, [])

  // Fetch metrics and predictions when selection changes
  useEffect(() => {
    if (!selectedGranularity || !selectedModel) return

    setLoading(true)
    setError(null)

    Promise.all([
      fetch(`${API_BASE}/metrics?granularity=${selectedGranularity}&model=${selectedModel}&horizon=${selectedHorizon}`)
        .then(r => r.ok ? r.json() : Promise.reject('Not found')),
      fetch(`${API_BASE}/predict?granularity=${selectedGranularity}&model=${selectedModel}&horizon=${selectedHorizon}`)
        .then(r => r.ok ? r.json() : Promise.reject('Not found'))
    ])
      .then(([metricsData, predsData]) => {
        setMetrics(metricsData)
        setPredictions(predsData)
        setLoading(false)
      })
      .catch(err => {
        setMetrics(null)
        setPredictions(null)
        setError(`No data for ${selectedModel} at ${GRANULARITY_LABELS[selectedGranularity]} granularity`)
        setLoading(false)
      })
  }, [selectedGranularity, selectedModel, selectedHorizon])

  // Get available models for selected granularity
  const availableModels = available[selectedGranularity] || []

  // Update model and horizon when granularity changes
  useEffect(() => {
    if (availableModels.length > 0) {
      const firstModel = availableModels[0]
      setSelectedModel(firstModel.model)
      setSelectedHorizon(firstModel.horizon)
    }
  }, [selectedGranularity, available])

  // Format chart data
  const chartData = predictions?.series?.map(item => ({
    time: formatTime(item.t, selectedGranularity),
    actual: item.actual,
    predicted: item.predicted
  })) || []

  return (
    <div className="app">
      <header className="header">
        <h1>UK Electricity Demand Forecast</h1>
        <p className="subtitle">Multi-timeframe forecasting dashboard</p>
      </header>

      <div className="controls">
        <div className="control-group">
          <label>Granularity</label>
          <select
            value={selectedGranularity}
            onChange={e => setSelectedGranularity(e.target.value)}
          >
            {granularities.map(g => (
              <option key={g.code} value={g.code}>
                {g.name.charAt(0).toUpperCase() + g.name.slice(1)}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Model</label>
          <select
            value={selectedModel}
            onChange={e => {
              const model = availableModels.find(m => m.model === e.target.value)
              setSelectedModel(e.target.value)
              if (model) setSelectedHorizon(model.horizon)
            }}
          >
            {availableModels.map(m => (
              <option key={m.model} value={m.model}>
                {getModelName(m.model)}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Horizon</label>
          <div className="horizon-display">
            {selectedHorizon} {getHorizonUnit(selectedGranularity)}
          </div>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {loading && <div className="loading">Loading...</div>}

      {metrics && !loading && (
        <div className="metrics-grid">
          <MetricCard label="MAE" value={formatNumber(metrics.mae)} unit="MW" description="Mean Absolute Error" />
          <MetricCard label="RMSE" value={formatNumber(metrics.rmse)} unit="MW" description="Root Mean Square Error" />
          <MetricCard label="SMAPE" value={metrics.smape?.toFixed(2)} unit="%" description="Symmetric MAPE" />
          <MetricCard label="MAPE" value={metrics.mape?.toFixed(2)} unit="%" description="Mean Absolute % Error" />
        </div>
      )}

      {predictions && !loading && chartData.length > 0 && (
        <div className="chart-container">
          <h2>Predictions vs Actual</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="time"
                stroke="#9ca3af"
                tick={{ fill: '#9ca3af', fontSize: 12 }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis
                stroke="#9ca3af"
                tick={{ fill: '#9ca3af' }}
                tickFormatter={v => formatNumber(v)}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#f3f4f6' }}
                formatter={(value) => [formatNumber(value) + ' MW', '']}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="actual"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="Actual"
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
                name="Predicted"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {predictions && !loading && (
        <div className="table-container">
          <h2>Prediction Data</h2>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Actual (MW)</th>
                <th>Predicted (MW)</th>
                <th>Error (MW)</th>
                <th>Error %</th>
              </tr>
            </thead>
            <tbody>
              {predictions.series.map((item, i) => {
                const error = item.predicted - item.actual
                const errorPct = ((error / item.actual) * 100).toFixed(2)
                return (
                  <tr key={i}>
                    <td>{formatTime(item.t, selectedGranularity)}</td>
                    <td>{formatNumber(item.actual)}</td>
                    <td>{formatNumber(item.predicted)}</td>
                    <td className={error > 0 ? 'positive' : 'negative'}>
                      {error > 0 ? '+' : ''}{formatNumber(error)}
                    </td>
                    <td className={error > 0 ? 'positive' : 'negative'}>
                      {error > 0 ? '+' : ''}{errorPct}%
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value, unit, description }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {value} <span className="metric-unit">{unit}</span>
      </div>
      <div className="metric-description">{description}</div>
    </div>
  )
}

function getModelName(id) {
  const names = {
    xgb: 'XGBoost',
    rf: 'Random Forest',
    linear: 'Linear Regression',
    ebm: 'EBM'
  }
  return names[id] || id
}

function getHorizonUnit(granularity) {
  const units = {
    H: 'hours',
    D: 'days',
    W: 'weeks',
    M: 'months',
    Y: 'years'
  }
  return units[granularity] || 'periods'
}

function formatTime(isoString, granularity) {
  const date = new Date(isoString)
  switch (granularity) {
    case 'H':
      return date.toLocaleString('en-GB', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    case 'D':
      return date.toLocaleDateString('en-GB', { month: 'short', day: 'numeric' })
    case 'W':
      return `Week of ${date.toLocaleDateString('en-GB', { month: 'short', day: 'numeric' })}`
    case 'M':
      return date.toLocaleDateString('en-GB', { month: 'long', year: 'numeric' })
    case 'Y':
      return date.getFullYear().toString()
    default:
      return isoString
  }
}

function formatNumber(num) {
  if (num === undefined || num === null) return '-'
  if (Math.abs(num) >= 1000000) {
    return (num / 1000000).toFixed(2) + 'M'
  }
  if (Math.abs(num) >= 1000) {
    return (num / 1000).toFixed(1) + 'k'
  }
  return num.toFixed(0)
}

export default App
