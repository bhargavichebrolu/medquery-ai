import { Link, useLocation } from 'react-router'
import { findPage, PAGES } from '../navigation'
import { useSelectedModel } from '../state/modelSelection'

function Header() {
  const { pathname } = useLocation()
  const page = findPage(pathname)
  const { selectedModel } = useSelectedModel()

  return (
    <header className="app-header">
      <h1>{page ? page.label : 'Page not found'}</h1>
      <Link to={PAGES.modelLab.path} className="model-badge">
        {selectedModel ? `Model: ${selectedModel.id}` : 'No model selected'}
      </Link>
    </header>
  )
}

export default Header
