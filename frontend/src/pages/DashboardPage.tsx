import { Link } from 'react-router'
import StatCard from '../components/StatCard'
import { DATASET_FACTS } from '../data/researchDataset'
import { NAV_ITEMS, PAGES } from '../navigation'
import { useSelectedModel } from '../state/modelSelection'

// Every page except the Dashboard itself.
const EXPLORE_PAGES = NAV_ITEMS.filter((page) => page !== PAGES.dashboard)

const UPCOMING = [
  {
    title: 'Research results summary',
    text: 'Verified retrieval results will appear here once the Analytics page is built (PRD 1.5).',
  },
  {
    title: 'Recent experiments',
    text: 'Experiment tracking is planned for Part 4.',
  },
]

function DashboardPage() {
  const { selectedModel } = useSelectedModel()

  return (
    <>
      <p className="page-purpose">{PAGES.dashboard.purpose}</p>

      <section className="dashboard-section" aria-labelledby="dataset-heading">
        <h2 id="dataset-heading">Research dataset</h2>
        <p className="section-note">
          Facts about the original NHS dataset. These are not model performance results.
        </p>
        <div className="card-grid">
          {DATASET_FACTS.map((fact) => (
            <StatCard key={fact.label} {...fact} />
          ))}
        </div>
      </section>

      <section className="dashboard-section" aria-labelledby="model-heading">
        <h2 id="model-heading">Selected embedding model</h2>
        <div className="config-panel">
          {selectedModel ? (
            <>
              <p className="stat-value">{selectedModel.id}</p>
              <p className="stat-detail">
                <code>{selectedModel.huggingFaceId}</code> · {selectedModel.dimensions} dimensions
              </p>
              <Link to={PAGES.modelLab.path} className="text-link">
                Change in the Model Lab
              </Link>
            </>
          ) : (
            <>
              <p className="stat-value">No model selected</p>
              <Link to={PAGES.modelLab.path} className="text-link">
                Choose one in the Model Lab
              </Link>
            </>
          )}
        </div>
      </section>

      <section className="dashboard-section" aria-labelledby="explore-heading">
        <h2 id="explore-heading">Explore</h2>
        <div className="card-grid">
          {EXPLORE_PAGES.map((page) => (
            <Link key={page.path} to={page.path} className="explore-card">
              <span className="explore-title">{page.label}</span>
              <span className="explore-purpose">{page.purpose}</span>
              <span className="explore-status">
                {page.plannedIn ? `Planned: ${page.plannedIn}` : 'Available'}
              </span>
            </Link>
          ))}
        </div>
      </section>

      <section className="dashboard-section" aria-labelledby="upcoming-heading">
        <h2 id="upcoming-heading">Coming later</h2>
        <div className="card-grid">
          {UPCOMING.map((panel) => (
            <div key={panel.title} className="placeholder">
              <h3 className="placeholder-label">{panel.title}</h3>
              <p>{panel.text}</p>
            </div>
          ))}
        </div>
      </section>
    </>
  )
}

export default DashboardPage
