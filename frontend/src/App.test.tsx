import { fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'
import App from './App'

// MemoryRouter keeps the address in memory instead of the browser's address bar,
// so each test can start the app at any page.
function renderAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )
  return within(screen.getByRole('navigation', { name: 'Main' }))
}

describe('application navigation', () => {
  it('shows a sidebar link for each of the five pages', () => {
    const nav = renderAt('/')

    const labels = nav.getAllByRole('link').map((link) => link.textContent)
    expect(labels).toEqual(['Dashboard', 'Chat', 'Knowledge Base', 'Model Lab', 'Analytics'])
  })

  it('opens the Dashboard at / and marks its link as the current page', () => {
    const nav = renderAt('/')

    expect(screen.getByRole('heading', { level: 1, name: 'Dashboard' })).toBeInTheDocument()
    expect(nav.getByRole('link', { name: 'Dashboard' })).toHaveAttribute('aria-current', 'page')
    expect(nav.getByRole('link', { name: 'Analytics' })).not.toHaveAttribute('aria-current')
  })

  it('switches to Analytics when its link is clicked', () => {
    const nav = renderAt('/')

    fireEvent.click(nav.getByRole('link', { name: 'Analytics' }))

    expect(screen.getByRole('heading', { level: 1, name: 'Analytics' })).toBeInTheDocument()
    expect(nav.getByRole('link', { name: 'Analytics' })).toHaveAttribute('aria-current', 'page')
    expect(nav.getByRole('link', { name: 'Dashboard' })).not.toHaveAttribute('aria-current')
  })

  it('shows "Page not found" for an unknown address', () => {
    const nav = renderAt('/xyz')

    expect(screen.getByRole('heading', { level: 1, name: 'Page not found' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Go to the Dashboard' })).toBeInTheDocument()
    expect(nav.queryByRole('link', { current: 'page' })).not.toBeInTheDocument()
  })
})
