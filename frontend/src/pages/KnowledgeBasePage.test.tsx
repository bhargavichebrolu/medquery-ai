import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'
import App from '../App'

function renderAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )
}

function articleRows() {
  const table = within(screen.getByRole('region', { name: 'Articles' })).getByRole('table')
  return Array.from(table.querySelectorAll('tbody tr'))
}

const countLine = () => screen.getByText(/^Showing \d+ of \d+ articles/)

describe('Knowledge Base catalogue', () => {
  it('lists all 698 articles, 50 at a time', () => {
    renderAt('/knowledge-base')

    expect(screen.getByRole('heading', { name: 'Original research dataset (read-only)' })).toBeInTheDocument()
    expect(screen.getByText(/698 NHS articles split into 3,796 chunks/)).toBeInTheDocument()
    expect(articleRows()).toHaveLength(50)
    expect(countLine()).toHaveTextContent('Showing 50 of 698 articles')

    fireEvent.click(screen.getByRole('button', { name: 'Show 50 more' }))
    expect(articleRows()).toHaveLength(100)
    // When the whole suite runs in parallel this test has taken 4.5-5 s, close to the default
    // 5 s limit (alone it takes under 2 s), so this test alone gets 10 s.
  }, 10_000)

  it('searches titles and filters by evaluation use and classification', () => {
    renderAt('/knowledge-base')

    fireEvent.change(screen.getByRole('searchbox', { name: 'Search titles' }), { target: { value: 'asthma' } })
    expect(articleRows().map((row) => row.querySelector('a')!.textContent)).toContain('Asthma')
    expect(articleRows().every((row) => /asthma/i.test(row.textContent!))).toBe(true)

    fireEvent.change(screen.getByRole('searchbox', { name: 'Search titles' }), { target: { value: '' } })
    fireEvent.click(screen.getByRole('radio', { name: 'Used in evaluation' }))
    expect(countLine()).toHaveTextContent('Showing 8 of 8 articles (698 in total)')

    fireEvent.click(screen.getByRole('radio', { name: 'Contents page only' }))
    expect(countLine()).toHaveTextContent('Showing 50 of 107 articles (698 in total)')
  })

  it('reads its filters from the address and states the classification tentatively', () => {
    renderAt('/knowledge-base?show=evaluation&sort=chunks')

    expect(screen.getByRole('radio', { name: 'Used in evaluation' })).toBeChecked()
    expect(articleRows()[0].querySelector('a')!.textContent).toBe('Hives')

    const quality = within(screen.getByRole('region', { name: 'Dataset classification: contents pages' }))
    expect(quality.getByText(/107 articles are classified/)).toBeInTheDocument()
    expect(quality.getByText(/3 of the 8 evaluation articles are classified this way/)).toBeInTheDocument()
    expect(quality.getByText(/models disagreed on questions 7 and 9, but all ranked the relevant article first on question 10/)).toBeInTheDocument()
    expect(quality.getByText(/this small evaluation cannot show that/)).toBeInTheDocument()
  })
})

describe('Article detail page', () => {
  it('opens from the catalogue at its own address and shows verbatim chunk text', () => {
    renderAt('/knowledge-base?show=evaluation')

    fireEvent.click(screen.getByRole('link', { name: 'Asthma' }))

    expect(screen.getByRole('heading', { level: 1, name: 'Knowledge Base' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { level: 2, name: 'Asthma' })).toBeInTheDocument()
    const nav = within(screen.getByRole('navigation', { name: 'Main' }))
    expect(nav.getByRole('link', { name: 'Knowledge Base' })).toHaveAttribute('aria-current', 'page')

    expect(screen.getByRole('heading', { name: 'Chunks (14)' })).toBeInTheDocument()
    const chunks = document.querySelectorAll('.chunk-text')
    expect(chunks).toHaveLength(14)
    expect(chunks[0].textContent!.startsWith('# Asthma')).toBe(true)
    expect(screen.getByText('Question 1: What are the symptoms of asthma?')).toBeInTheDocument()

    // The back link returns to the catalogue with the same filter.
    expect(screen.getByRole('link', { name: '← All articles' })).toHaveAttribute(
      'href',
      '/knowledge-base?show=evaluation',
    )
  })

  it('shows lengths only for non-evaluation articles, and handles unknown articles', () => {
    renderAt('/knowledge-base/noise_sensitivity_hyperacusis')

    expect(screen.getByRole('heading', { level: 2, name: 'Noise sensitivity (hyperacusis)' })).toBeInTheDocument()
    expect(screen.getByText(/chunk text will be available when the knowledge base is connected to the backend in Part 2/)).toBeInTheDocument()
    expect(document.querySelectorAll('.chunk-text')).toHaveLength(0)
    expect(screen.getByText('Not used by the evaluation questions')).toBeInTheDocument()
  })

  it('shows a not-found message for an unknown article and never links to NHS pages', () => {
    renderAt('/knowledge-base/not-a-real-article')

    expect(screen.getByRole('heading', { level: 2, name: 'Article not found' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { level: 1, name: 'Knowledge Base' })).toBeInTheDocument()

    // Every article page with text, plus the catalogue: no link leaves the app, no nhs.uk address appears.
    for (const path of ['/knowledge-base', '/knowledge-base/hives', '/knowledge-base/amoxicillin']) {
      cleanup()
      renderAt(path)
      const hrefs = Array.from(document.querySelectorAll('a')).map((a) => a.getAttribute('href') ?? '')
      expect(hrefs.every((href) => href.startsWith('/')), path).toBe(true)
      expect(document.querySelector('.chunk-text a'), path).toBeNull()
      expect(document.body.textContent, path).not.toMatch(/https?:\/\/[^\s]*nhs\.uk|www\.nhs\.uk/i)
    }
  })

  it('says when verbatim chunk text contains web addresses, shown as plain text', () => {
    renderAt('/knowledge-base/hives')

    expect(screen.getByText(/shown as plain text, not links/)).toBeInTheDocument()
  })
})
