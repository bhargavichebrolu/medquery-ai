import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import BackendStatus from './BackendStatus'

const REAL_COUNTS = {
  articles: 698,
  chunks: 3796,
  embeddingIndexes: ['BGE', 'E5', 'MPNet', 'MiniLM'],
  source: 'Healthcare_RAG/data',
}

function respondWith(body: unknown, status = 200) {
  const fetchMock = vi.fn(() =>
    Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })),
  )
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('Backend connection panel', () => {
  it('shows a loading state, then the live counts and that they match', async () => {
    const fetchMock = respondWith(REAL_COUNTS)
    render(<BackendStatus />)

    expect(screen.getByText('Checking the backend…')).toBeInTheDocument()
    expect(await screen.findByText('Connected to the MedQuery backend')).toBeInTheDocument()
    expect(screen.getByText(/articles ·/).textContent).toBe(
      '698 articles · 3,796 chunks · 4 embedding indexes (BGE, E5, MPNet, MiniLM)',
    )
    expect(screen.getByText('Matches the verified dataset figures above.')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith('/api/dataset/summary', expect.anything())
  })

  it('explains how to start the backend when it cannot be reached, and can try again', async () => {
    const fetchMock = vi.fn<() => Promise<Response>>(() => Promise.reject(new TypeError('Failed to fetch')))
    vi.stubGlobal('fetch', fetchMock)
    render(<BackendStatus />)

    expect(await screen.findByText('Backend not available')).toBeInTheDocument()
    expect(screen.getByText('Failed to fetch')).toBeInTheDocument()
    expect(screen.getByText(/uv run uvicorn medquery_api\.main:app --reload --host 127\.0\.0\.1 --port 8000/)).toBeInTheDocument()

    fetchMock.mockImplementation(() =>
      Promise.resolve(new Response(JSON.stringify(REAL_COUNTS), { status: 200 })),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByText('Connected to the MedQuery backend')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('says the backend could not be reached when the development proxy answers 502', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response('', { status: 502 }))))
    render(<BackendStatus />)

    expect(
      await screen.findByText('Could not reach the backend (HTTP 502 from the development proxy).'),
    ).toBeInTheDocument()
  })

  it("shows the backend's error message when it answers with an error", async () => {
    respondWith({ detail: 'Research data folder not found. Restore Healthcare_RAG/data or set MEDQUERY_DATA_DIR.' }, 503)
    render(<BackendStatus />)

    expect(await screen.findByText(/HTTP 503\. Research data folder not found/)).toBeInTheDocument()
  })

  it('warns, without replacing anything, when the counts differ from the verified figures', async () => {
    respondWith({ ...REAL_COUNTS, articles: 690, embeddingIndexes: ['BGE', 'E5'] })
    render(<BackendStatus />)

    const warning = await screen.findByRole('alert')
    expect(warning).toHaveTextContent("The backend's counts differ from the verified figures")
    expect(warning).toHaveTextContent('Articles: 690 (verified: 698)')
    expect(warning).toHaveTextContent('Embedding indexes: BGE, E5 (verified: BGE, E5, MPNet, MiniLM)')
    expect(warning).not.toHaveTextContent('Chunks:')
  })

  it('rejects a response in an unexpected format', async () => {
    respondWith({ articles: 'many' })
    render(<BackendStatus />)

    expect(await screen.findByText('The backend sent a response in an unexpected format.')).toBeInTheDocument()
  })
})
