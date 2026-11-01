import { Outlet } from 'react-router'
import Header from './Header'
import Sidebar from './Sidebar'

// The frame shared by every page. <Outlet /> is the slot where React Router
// places the page that matches the current address.
function AppLayout() {
  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main-column">
        <Header />
        <main className="page">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default AppLayout
