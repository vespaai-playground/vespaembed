import { useRef, useEffect, useMemo } from 'react'
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Filler,
  Tooltip,
  type ChartData,
  type ChartOptions,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import type { MetricPoint } from '../../context/AppContext'

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Filler, Tooltip)

interface MetricsChartProps {
  metricsData: Record<string, MetricPoint[]>
  currentMetric: string
  availableMetrics: string[]
  onMetricChange: (metric: string) => void
  isTraining?: boolean
}

export function MetricsChart({ metricsData, currentMetric, availableMetrics, onMetricChange, isTraining }: MetricsChartProps) {
  const chartRef = useRef<ChartJS<'line'>>(null)

  const metricData = metricsData[currentMetric]
  const hasData = metricData && metricData.length > 0

  const chartData: ChartData<'line'> = useMemo(() => {
    if (!hasData) {
      return { labels: [], datasets: [{ label: 'Loss', data: [] }] }
    }
    return {
      labels: metricData.map(d => d.step),
      datasets: [{
        label: currentMetric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        data: metricData.map(d => d.value),
        borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointRadius: 3,
        pointBackgroundColor: '#22c55e',
        pointBorderColor: '#22c55e',
        pointHoverRadius: 6,
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: '#22c55e',
        pointHoverBorderWidth: 2,
      }]
    }
  }, [hasData, metricData, currentMetric])

  const chartOptions: ChartOptions<'line'> = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false,
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: '#22c55e',
        borderWidth: 1,
        displayColors: false,
        callbacks: {
          title: (items) => `Step ${items[0].label}`,
          label: (item) => `${item.dataset.label}: ${Number(item.parsed.y).toFixed(4)}`,
        }
      }
    },
    scales: {
      x: {
        display: true,
        grid: { color: 'rgba(255,255,255,0.05)' },
        ticks: { color: '#999' }
      },
      y: {
        display: true,
        grid: { color: 'rgba(255,255,255,0.05)' },
        ticks: { color: '#999' }
      }
    }
  }), [])

  // Force chart update when data changes
  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.update('none')
    }
  }, [chartData])

  const displayMetrics = availableMetrics.length > 0
    ? availableMetrics
    : ['loss']

  return (
    <div className="chart-card">
      <div className="chart-header">
        <span>Training Progress</span>
        <div className="chart-controls">
          <select
            className="metric-select"
            value={currentMetric}
            onChange={(e) => onMetricChange(e.target.value)}
          >
            {displayMetrics.map(metric => (
              <option key={metric} value={metric}>
                {metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="chart-wrapper">
        <Line ref={chartRef} data={chartData} options={chartOptions} />
        {!hasData && (
          <div className="chart-placeholder" style={{ display: 'flex' }}>
            {isTraining ? 'Waiting for metrics...' : 'Start training to see progress'}
          </div>
        )}
      </div>
    </div>
  )
}
