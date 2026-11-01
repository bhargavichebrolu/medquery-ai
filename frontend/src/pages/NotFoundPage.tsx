import { Link, useLocation } from 'react-router'
import { PAGES } from '../navigation'

function NotFoundPage() {
  const { pathname } = useLocation()

  return (
    <>
      <p className="page-purpose">
        There is no page at <code>{pathname}</code>.
      </p>
      <Link to={PAGES.dashboard.path} className="text-link">
        Go to the Dashboard
      </Link>
    </>
  )
}

export default NotFoundPage
