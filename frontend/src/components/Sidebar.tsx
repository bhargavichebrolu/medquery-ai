import { NavLink } from 'react-router'
import { NAV_ITEMS } from '../navigation'

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          M
        </span>
        <div>
          <p className="brand-name">MedQuery AI</p>
          <p className="brand-tagline">RAG Experiment Studio</p>
        </div>
      </div>

      <nav aria-label="Main">
        <ul className="nav-list">
          {NAV_ITEMS.map((page) => (
            <li key={page.path}>
              {/* NavLink sets aria-current="page" on the link for the current address. */}
              <NavLink to={page.path} className="nav-link">
                {page.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      <p className="sidebar-footer">Part 1 · frontend</p>
    </aside>
  )
}

export default Sidebar
