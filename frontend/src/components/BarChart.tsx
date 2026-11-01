import { useId, type CSSProperties } from 'react'

export interface Bar {
  label: string
  value: number
  displayValue: string
  exactValue: string
  emphasised: boolean
}

interface BarChartProps {
  title: string
  subtitle: string
  bars: Bar[]
  max: number
}

// A horizontal bar chart for a single measure. Each bar shows its value at the tip;
// hovering or focusing a row shows the exact, unrounded value. When one bar is
// emphasised the others turn grey, so the chosen model stands out.
function BarChart({ title, subtitle, bars, max }: BarChartProps) {
  const anyEmphasised = bars.some((bar) => bar.emphasised)
  const titleId = useId()
  const subtitleId = useId()

  return (
    <figure className="bar-chart" aria-labelledby={titleId} aria-describedby={subtitleId}>
      <figcaption>
        <span id={titleId} className="chart-title">
          {title}
        </span>
        <span id={subtitleId} className="chart-subtitle">
          {subtitle}
        </span>
      </figcaption>
      <ul className="bar-list">
        {bars.map((bar) => (
          <li
            key={bar.label}
            className="bar-row"
            data-muted={anyEmphasised && !bar.emphasised ? 'true' : undefined}
            tabIndex={0}
          >
            <span className="bar-label">{bar.label}</span>
            {/* --share is the bar's length as a percentage; CSS uses it for the
                bar width and to place the value just past the bar's tip. */}
            <span
              className="bar-area"
              style={{ '--share': `${(bar.value / max) * 100}%` } as CSSProperties}
            >
              <span className="bar-fill" />
              <span className="bar-value">{bar.displayValue}</span>
            </span>
            <span className="bar-tooltip" aria-hidden="true">
              <strong>{bar.exactValue}</strong> {bar.label}, exact value
            </span>
          </li>
        ))}
      </ul>
    </figure>
  )
}

export default BarChart
