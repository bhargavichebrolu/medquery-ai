interface StatCardProps {
  label: string
  value: string
  detail: string
  source: string
}

function StatCard({ label, value, detail, source }: StatCardProps) {
  return (
    <div className="stat-card">
      <p className="stat-label">{label}</p>
      <p className="stat-value">{value}</p>
      <p className="stat-detail">{detail}</p>
      <p className="stat-source">Source: {source}</p>
    </div>
  )
}

export default StatCard
