import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'

describe('App', () => {
  it('shows the MedQuery AI title and subtitle', () => {
    render(<App />)

    expect(screen.getByRole('heading', { level: 1, name: 'MedQuery AI' })).toBeInTheDocument()
    expect(screen.getByText('RAG Experiment Studio')).toBeInTheDocument()
  })
})
