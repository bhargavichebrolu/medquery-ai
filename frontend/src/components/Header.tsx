import { useLocation } from 'react-router'
import { findPage } from '../navigation'

function Header() {
  const { pathname } = useLocation()
  const page = findPage(pathname)

  return (
    <header className="app-header">
      <h1>{page ? page.label : 'Page not found'}</h1>
    </header>
  )
}

export default Header
