import { fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'
import App from '../App'

function renderDashboard() {
  render(
    <MemoryRouter initialEntries={['/']}>
      <App />
    </MemoryRouter>,
  )
}

// A <section> labelled by a heading is exposed as a "region" with that heading's name.
function section(name: string) {
  return within(screen.getByRole('region', { name }))
}

describe('Dashboard', () => {
  it('shows the four verified dataset facts, each citing the research dataset', () => {
    renderDashboard()
    const dataset = section('Research dataset')

    for (const text of [
      'NHS articles',
      '698',
      'Text chunks',
      '3,796',
      'Embedding models',
      '4',
      'MPNet, MiniLM, BGE, E5',
      'Chunking',
      '1,000 characters',
      '200-character overlap between chunks',
    ]) {
      expect(dataset.getByText(text)).toBeInTheDocument()
    }
    expect(dataset.getAllByText('Source: original research dataset')).toHaveLength(4)
  })

  it('links to the five other areas of the application', () => {
    renderDashboard()

    const hrefs = section('Explore')
      .getAllByRole('link')
      .map((link) => link.getAttribute('href'))
    expect(hrefs).toEqual(['/chat', '/retrieval', '/knowledge-base', '/model-lab', '/analytics'])
  })

  it('opens the Model Lab when its card is clicked', () => {
    renderDashboard()

    fireEvent.click(section('Explore').getByRole('link', { name: /^Model Lab/ }))

    expect(screen.getByRole('heading', { level: 1, name: 'Model Lab' })).toBeInTheDocument()
  })

  it('links to research results without showing any performance metrics', () => {
    renderDashboard()
    const research = section('Research and experiments')

    for (const title of ['Research results', 'Recent experiments']) {
      expect(research.getByRole('heading', { name: title })).toBeInTheDocument()
    }
    expect(research.getByRole('link', { name: 'View in Analytics' })).toHaveAttribute('href', '/analytics')
    expect(document.body.textContent).not.toMatch(/MRR|nDCG|precision|recall/i)
  })
})
