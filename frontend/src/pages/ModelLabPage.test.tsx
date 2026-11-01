import { fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { describe, expect, it } from 'vitest'
import App from '../App'
import { STORAGE_KEY } from '../state/modelSelection'

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )
}

function radio(modelId: string) {
  return screen.getByRole('radio', { name: modelId })
}

function section(name: string | RegExp) {
  return within(screen.getByRole('region', { name }))
}

const VERIFIED_MODELS = [
  { id: 'MPNet', huggingFaceId: 'sentence-transformers/all-mpnet-base-v2', dimensions: '768' },
  { id: 'MiniLM', huggingFaceId: 'sentence-transformers/all-MiniLM-L6-v2', dimensions: '384' },
  { id: 'BGE', huggingFaceId: 'BAAI/bge-small-en-v1.5', dimensions: '384' },
  { id: 'E5', huggingFaceId: 'intfloat/multilingual-e5-base', dimensions: '768' },
]

describe('Model Lab', () => {
  it('lists the four models, none selected, with no performance metrics', () => {
    renderAt('/model-lab')

    for (const model of VERIFIED_MODELS) {
      expect(radio(model.id)).not.toBeChecked()
      expect(radio(model.id)).toHaveAccessibleDescription(`${model.dimensions} dimensions`)
    }
    expect(screen.getByRole('banner')).toHaveTextContent('No model selected')
    expect(screen.getByText('Choose a model above to see its configuration.')).toBeInTheDocument()
    expect(document.body.textContent).not.toMatch(/MRR|nDCG|precision|recall/i)
  })

  it("shows each selected model's verified configuration", () => {
    renderAt('/model-lab')

    for (const model of VERIFIED_MODELS) {
      fireEvent.click(radio(model.id))

      expect(radio(model.id)).toBeChecked()
      const config = section(`Selected configuration: ${model.id}`)
      expect(config.getByText(model.huggingFaceId)).toBeInTheDocument()
      expect(config.getByText(model.dimensions)).toBeInTheDocument()
    }

    // Only E5 uses text prefixes.
    expect(section(/Selected configuration/).getByText('query:')).toBeInTheDocument()
    fireEvent.click(radio('BGE'))
    expect(section(/Selected configuration/).getByText('None (text is embedded as it is)')).toBeInTheDocument()
  })

  it('shows the selected model in the header and on the Dashboard', () => {
    renderAt('/model-lab')

    fireEvent.click(radio('BGE'))
    expect(screen.getByRole('banner')).toHaveTextContent('Model: BGE')

    const nav = within(screen.getByRole('navigation', { name: 'Main' }))
    fireEvent.click(nav.getByRole('link', { name: 'Dashboard' }))

    expect(screen.getByRole('heading', { level: 1, name: 'Dashboard' })).toBeInTheDocument()
    expect(screen.getByRole('banner')).toHaveTextContent('Model: BGE')
    const dashboardPanel = section('Selected embedding model')
    expect(dashboardPanel.getByText('BGE')).toBeInTheDocument()
    expect(dashboardPanel.getByText('BAAI/bge-small-en-v1.5')).toBeInTheDocument()
  })

  it('remembers the choice after a reload and ignores invalid saved values', () => {
    const firstVisit = renderAt('/model-lab')
    fireEvent.click(radio('E5'))
    expect(localStorage.getItem(STORAGE_KEY)).toBe('E5')
    firstVisit.unmount()

    // A fresh render reads the saved choice, as a page refresh would.
    const secondVisit = renderAt('/model-lab')
    expect(radio('E5')).toBeChecked()
    expect(screen.getByRole('banner')).toHaveTextContent('Model: E5')
    secondVisit.unmount()

    localStorage.setItem(STORAGE_KEY, 'not-a-real-model')
    renderAt('/model-lab')
    for (const model of VERIFIED_MODELS) {
      expect(radio(model.id)).not.toBeChecked()
    }
    expect(screen.getByRole('banner')).toHaveTextContent('No model selected')
  })
})
