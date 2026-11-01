import type { PageInfo } from '../navigation'

interface PlaceholderPageProps {
  page: PageInfo
}

function PlaceholderPage({ page }: PlaceholderPageProps) {
  return (
    <>
      <p className="page-purpose">{page.purpose}</p>
      <div className="placeholder">
        <p className="placeholder-label">Placeholder</p>
        <p>This page will be built in {page.plannedIn ?? 'a later milestone'}.</p>
      </div>
    </>
  )
}

export default PlaceholderPage
