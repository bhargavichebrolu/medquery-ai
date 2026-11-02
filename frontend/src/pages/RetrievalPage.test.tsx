import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, useLocation, useNavigate } from 'react-router'
import { describe, expect, it, vi } from 'vitest'
import App from '../App'
import type { RetrievalResponse } from '../api'
import { ARTICLES } from '../data/knowledgeBase'
import { STORAGE_KEY } from '../state/modelSelection'

// Shows the current address, and offers a Back button, so tests can follow the history.
function HistoryTools() {
  const location = useLocation()
  const navigate = useNavigate()
  return (
    <>
      <span data-testid="address">{location.pathname + location.search}</span>
      <button type="button" onClick={() => navigate(-1)}>
        Test back
      </button>
    </>
  )
}

function renderAt(path: string, savedModel?: string) {
  if (savedModel) {
    localStorage.setItem(STORAGE_KEY, savedModel)
  }
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
      <HistoryTools />
    </MemoryRouter>,
  )
}

const address = () => screen.getByTestId('address').textContent
const questionBox = () => screen.getByRole('textbox', { name: 'Question' }) as HTMLTextAreaElement
const retrieveButton = () => screen.getByRole('button', { name: 'Retrieve' })
const status = () => screen.getByRole('status')

function ask(question: string, topK?: number) {
  fireEvent.change(questionBox(), { target: { value: question } })
  if (topK) {
    fireEvent.change(screen.getByRole('combobox', { name: 'Number of results' }), {
      target: { value: String(topK) },
    })
  }
  fireEvent.click(retrieveButton())
}

const ASTHMA_TEXT = '# Asthma\n\nAsthma is a common condition.\n\n• wheezing\n\nSee https://www.example.org/help for more.'
const SHARED_TEXT = '## Immediate action required: Call 999 if:\n\n• your lips suddenly become swollen'

function response(overrides: Partial<RetrievalResponse> = {}): RetrievalResponse {
  return {
    question: 'What are the symptoms of asthma?',
    model: 'BGE',
    modelId: 'BAAI/bge-small-en-v1.5',
    revision: '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a',
    topK: 4,
    results: [
      { rank: 1, score: 0.8265177011489868, sourceFile: 'asthma.txt', chunkId: 1, chunkLength: ASTHMA_TEXT.length, text: ASTHMA_TEXT },
      { rank: 2, score: 0.8026978969573975, sourceFile: 'asthma.txt', chunkId: 12, chunkLength: 10, text: 'Chunk 12.' },
      { rank: 3, score: 0.7466700077056885, sourceFile: 'food_allergy.txt', chunkId: 2, chunkLength: SHARED_TEXT.length, text: SHARED_TEXT },
      { rank: 4, score: 0.7466700077056885, sourceFile: 'hives.txt', chunkId: 13, chunkLength: SHARED_TEXT.length, text: SHARED_TEXT },
    ],
    timings: { modelLoadMs: 2699.4, embeddingMs: 8.6, searchMs: 0.58 },
    ...overrides,
  }
}

function respondWith(body: unknown, init: ResponseInit = { status: 200 }) {
  const fetchMock = vi.fn<(url: string, init: RequestInit) => Promise<Response>>(() =>
    Promise.resolve(new Response(typeof body === 'string' ? body : JSON.stringify(body), init)),
  )
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

// Requests that answer only when the test says so, to control their order.
function controlledFetch() {
  const calls: { body: unknown; signal: AbortSignal; answer: (body: unknown) => void }[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(
      (_url: string, init: RequestInit) =>
        new Promise<Response>((resolve) => {
          calls.push({
            body: JSON.parse(String(init.body)),
            signal: init.signal as AbortSignal,
            answer: (body) => resolve(new Response(JSON.stringify(body), { status: 200 })),
          })
        }),
    ),
  )
  return calls
}

describe('Retrieval form', () => {
  it('asks for a model and keeps Retrieve disabled until a question and model are given', () => {
    renderAt('/retrieval')

    expect(screen.getByRole('group', { name: 'Embedding model' })).toHaveTextContent('Choose a model to search with.')
    expect(retrieveButton()).toBeDisabled()

    fireEvent.change(questionBox(), { target: { value: 'What causes migraines?' } })
    expect(retrieveButton()).toBeDisabled()

    fireEvent.click(screen.getByRole('radio', { name: 'MPNet' }))
    expect(retrieveButton()).toBeEnabled()
    expect(screen.getByRole('link', { name: 'Model: MPNet' })).toBeInTheDocument() // the header badge
    expect(localStorage.getItem(STORAGE_KEY)).toBe('MPNet')
  })

  it('counts characters after trimming and refuses more than 1,000', () => {
    renderAt('/retrieval', 'BGE')

    fireEvent.change(questionBox(), { target: { value: `   ${'x'.repeat(1000)}   ` } })
    expect(screen.getByText('1,000 / 1,000 characters')).toBeInTheDocument()
    expect(retrieveButton()).toBeEnabled()

    fireEvent.change(questionBox(), { target: { value: 'x'.repeat(1001) } })
    expect(screen.getByText('1,001 / 1,000 characters')).toBeInTheDocument()
    expect(questionBox()).toHaveAttribute('aria-invalid', 'true')
    expect(retrieveButton()).toBeDisabled()
    expect(retrieveButton()).toHaveAccessibleDescription('Shorten the question to 1,000 characters or fewer.')

    fireEvent.change(questionBox(), { target: { value: '   ' } })
    expect(retrieveButton()).toHaveAccessibleDescription('Enter a question.')
  })

  it('offers 1 to 20 results, 5 by default', () => {
    renderAt('/retrieval', 'BGE')
    const select = screen.getByRole('combobox', { name: 'Number of results' }) as HTMLSelectElement

    expect(select.value).toBe('5')
    expect(Array.from(select.options).map((o) => o.value)).toEqual(
      Array.from({ length: 20 }, (_, i) => String(i + 1)),
    )
  })

  it('fills the question from an example without searching', () => {
    const fetchMock = respondWith(response())
    renderAt('/retrieval', 'BGE')

    fireEvent.click(screen.getByText('Example questions from the research evaluation'))
    fireEvent.click(screen.getByRole('button', { name: 'What are the symptoms of hives?' }))

    expect(questionBox().value).toBe('What are the symptoms of hives?')
    expect(questionBox()).toHaveFocus()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('Retrieval results', () => {
  it('sends the trimmed question, model and number of results, and records the search in the address', async () => {
    const fetchMock = respondWith(response())
    renderAt('/retrieval', 'BGE')

    ask('  What are the symptoms of asthma?  ', 4)

    await screen.findByRole('heading', { name: 'Retrieved chunks' })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/retrieve')
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({
      question: 'What are the symptoms of asthma?',
      model: 'BGE',
      topK: 4,
    })
    expect(address()).toBe('/retrieval?q=What+are+the+symptoms+of+asthma%3F&model=BGE&k=4')
    expect(status()).toHaveTextContent('Found 4 chunks with BGE.')
  })

  it('shows each chunk with its rank, score, article and verbatim text', async () => {
    respondWith(response())
    renderAt('/retrieval?q=What+are+the+symptoms+of+asthma%3F&model=BGE&k=4')

    const results = within(await screen.findByRole('region', { name: 'Retrieved chunks' }))
    const cards = results.getAllByRole('listitem')
    expect(cards).toHaveLength(4)

    const first = within(cards[0])
    expect(cards[0]).toHaveTextContent(/^Rank 1/)
    expect(first.getByRole('link', { name: 'Asthma' })).toHaveAttribute('href', '/knowledge-base/asthma')
    const score = cards[0].querySelector('data')!
    expect(score.textContent).toBe('0.8265')
    expect(score).toHaveAttribute('value', '0.8265177011489868')
    expect(score).toHaveAttribute('title', 'Exact score: 0.8265177011489868')
    expect(cards[0]).toHaveTextContent('(exact value 0.8265177011489868)')
    const asthmaChunks = ARTICLES.find((a) => a.id === 'asthma')!.chunkLengths.length
    expect(cards[0]).toHaveTextContent(`asthma.txt · chunk 1 of ${asthmaChunks}`)

    // Verbatim, line breaks kept, and the web address is plain text, not a link.
    expect(cards[0].querySelector('.chunk-text')!.textContent).toBe(ASTHMA_TEXT)
    expect(first.getAllByRole('link')).toHaveLength(1)
    expect(document.querySelector('a[href*="example.org"]')).toBeNull()
  })

  it('marks only exactly equal scores as tied, with a careful explanation', async () => {
    respondWith(response())
    renderAt('/retrieval?q=What+are+the+symptoms+of+asthma%3F&model=BGE&k=4')

    const region = await screen.findByRole('region', { name: 'Retrieved chunks' })
    const cards = within(region).getAllByRole('listitem')
    expect(cards.map((card) => within(card).queryByText('Tied score') !== null)).toEqual([false, false, true, true])
    expect(region).toHaveTextContent(
      'Results marked “Tied score” have exactly equal scores and are listed in index order; ' +
        'the original research may have listed tied results in a different order.',
    )
    expect(region).toHaveTextContent('They can be compared within one model, not between models.')
  })

  it('shows timings, mentioning model loading only when it happened', async () => {
    respondWith(response({ timings: { modelLoadMs: 0, embeddingMs: 8.6, searchMs: 0.58 } }))
    renderAt('/retrieval?q=asthma&model=BGE&k=4')

    const region = await screen.findByRole('region', { name: 'Retrieved chunks' })
    expect(region).toHaveTextContent('Question embedded in 8.6 ms · searched in 0.6 ms')
    expect(region).not.toHaveTextContent('Model loaded')
  })

  it('says when the model had to be loaded', async () => {
    respondWith(response())
    renderAt('/retrieval?q=asthma&model=BGE&k=4')

    expect(await screen.findByRole('region', { name: 'Retrieved chunks' })).toHaveTextContent('Model loaded in 2.7 s')
  })

  it('shows a loading message while waiting', () => {
    renderAt('/retrieval?q=asthma&model=E5&k=5') // the default fetch never answers

    expect(screen.getByText('Retrieving…')).toBeInTheDocument()
    expect(status()).toHaveTextContent('Retrieving chunks with E5…')
  })
})

describe('Retrieval errors', () => {
  it('explains how to start the backend when it cannot be reached, and can try again', async () => {
    const fetchMock = vi.fn<() => Promise<Response>>(() => Promise.reject(new TypeError('Failed to fetch')))
    vi.stubGlobal('fetch', fetchMock)
    renderAt('/retrieval?q=asthma&model=BGE&k=5')

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Backend not available')
    expect(alert).toHaveTextContent('Could not reach the backend.')
    expect(alert).toHaveTextContent('uv run uvicorn medquery_api.main:app --reload --host 127.0.0.1 --port 8000')

    fetchMock.mockImplementation(() => Promise.resolve(new Response(JSON.stringify(response()), { status: 200 })))
    fireEvent.click(within(alert).getByRole('button', { name: 'Try again' }))
    expect(await screen.findByRole('heading', { name: 'Retrieved chunks' })).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it.each([
    [502, '', 'Could not reach the backend (HTTP 502 from the development proxy).'],
    [504, '', 'Could not reach the backend (HTTP 504 from the development proxy).'],
    [
      503,
      { detail: 'E5 is not in the local model cache. Download it with: uv run python -m medquery_api.embeddings download E5' },
      'E5 is not in the local model cache. Download it with: uv run python -m medquery_api.embeddings download E5',
    ],
    [503, { detail: { nested: true } }, 'The backend cannot retrieve chunks right now.'],
    [
      422,
      { detail: [{ loc: ['body', 'question'], msg: 'String should have at least 1 character', type: 'string_too_short' }] },
      'The backend rejected the request: Question: String should have at least 1 character.',
    ],
    [422, 'not json', 'The backend rejected the request as invalid.'],
    [500, 'Internal Server Error', 'The backend answered with HTTP 500.'],
    [200, { results: 'many' }, 'The backend sent a response in an unexpected format.'],
    [200, 'not json', 'The backend sent a response in an unexpected format.'],
  ])('HTTP %i (%j) shows: %s', async (statusCode, body, message) => {
    respondWith(body, { status: statusCode })
    renderAt('/retrieval?q=asthma&model=E5&k=5')

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(message)
    expect(alert.textContent).not.toMatch(/Traceback|\\Users\\|\/home\//)
    expect(screen.queryByRole('heading', { name: 'Retrieved chunks' })).not.toBeInTheDocument()
  })

  it('never shows the text of an unexpected error', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ status: 200, ok: true, json: () => { throw new RangeError('C:\\secret\\path') } })))
    renderAt('/retrieval?q=asthma&model=E5&k=5')

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('The backend sent a response in an unexpected format.')
    expect(alert).not.toHaveTextContent('secret')
  })
})

describe('Retrieval address and requests', () => {
  it('fills the form from a complete address, selects its model everywhere, and searches once', async () => {
    const fetchMock = respondWith(response({ model: 'E5', modelId: 'intfloat/multilingual-e5-base', topK: 7 }))
    renderAt('/retrieval?q=What+is+aciclovir+used+to+treat%3F&model=E5&k=7', 'BGE')

    await screen.findByRole('heading', { name: 'Retrieved chunks' })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(JSON.parse(String(fetchMock.mock.calls[0][1].body))).toEqual({
      question: 'What is aciclovir used to treat?',
      model: 'E5',
      topK: 7,
    })
    expect(questionBox().value).toBe('What is aciclovir used to treat?')
    expect(screen.getByRole('radio', { name: 'E5' })).toBeChecked()
    expect(screen.getByRole('link', { name: 'Model: E5' })).toBeInTheDocument()
    expect(localStorage.getItem(STORAGE_KEY)).toBe('E5')
  })

  it.each([
    ['/retrieval?q=hello', 'hello', null, '5'],
    ['/retrieval?q=hello&model=GPT&k=5', 'hello', null, '5'],
    ['/retrieval?q=hello&model=bge&k=5', 'hello', null, '5'],
    ['/retrieval?q=hello&model=BGE&k=50', 'hello', 'BGE', '5'],
    ['/retrieval?q=hello&model=BGE&k=2.5', 'hello', 'BGE', '5'],
    ['/retrieval?q=hello&model=BGE', 'hello', 'BGE', '5'],
    ['/retrieval?q=%20%20&model=BGE&k=3', '  ', 'BGE', '3'],
    [`/retrieval?q=${'x'.repeat(1001)}&model=MiniLM&k=3`, 'x'.repeat(1001), 'MiniLM', '3'],
    ['/retrieval?model=MPNet&k=3', '', 'MPNet', '3'],
  ])('%s fills what is safe and does not search', (path, question, model, topK) => {
    const fetchMock = respondWith(response())
    renderAt(path)

    expect(fetchMock).not.toHaveBeenCalled()
    expect(questionBox().value).toBe(question)
    expect((screen.getByRole('combobox', { name: 'Number of results' }) as HTMLSelectElement).value).toBe(topK)
    for (const name of ['MPNet', 'MiniLM', 'BGE', 'E5']) {
      expect(screen.getByRole('radio', { name })).toHaveProperty('checked', name === model)
    }
    expect(screen.queryByText('Retrieving…')).not.toBeInTheDocument()
  })

  it('aborts an older search when a new one starts, and ignores its late answer', async () => {
    const calls = controlledFetch()
    renderAt('/retrieval', 'BGE')

    ask('What are the symptoms of asthma?')
    ask('What are the symptoms of hives?')
    expect(calls).toHaveLength(2)
    expect(calls[0].signal.aborted).toBe(true)
    expect(calls[1].signal.aborted).toBe(false)

    await act(async () => calls[1].answer(response({ question: 'What are the symptoms of hives?' })))
    await act(async () => calls[0].answer(response({ question: 'What are the symptoms of asthma?' })))

    const region = screen.getByRole('region', { name: 'Retrieved chunks' })
    expect(region).toHaveTextContent('“What are the symptoms of hives?”')
    expect(region).not.toHaveTextContent('asthma?”')
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('aborts the request when leaving the page, without showing an error', async () => {
    const calls = controlledFetch()
    renderAt('/retrieval?q=asthma&model=BGE&k=5')

    fireEvent.click(within(screen.getByRole('navigation', { name: 'Main' })).getByRole('link', { name: 'Dashboard' }))

    expect(calls[0].signal.aborted).toBe(true)
    await act(async () => calls[0].answer(response()))
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('goes back to the previous search with Back', async () => {
    const fetchMock = vi.fn((_url: string, init: RequestInit) => {
      const { question } = JSON.parse(String(init.body))
      return Promise.resolve(new Response(JSON.stringify(response({ question })), { status: 200 }))
    })
    vi.stubGlobal('fetch', fetchMock)
    renderAt('/retrieval', 'BGE')

    ask('What are the symptoms of asthma?')
    await screen.findByText(/“What are the symptoms of asthma\?”/)
    ask('What are the symptoms of hives?')
    await screen.findByText(/“What are the symptoms of hives\?”/)

    fireEvent.click(screen.getByRole('button', { name: 'Test back' }))

    expect(await screen.findByText(/“What are the symptoms of asthma\?”/)).toBeInTheDocument()
    expect(address()).toBe('/retrieval?q=What+are+the+symptoms+of+asthma%3F&model=BGE&k=5')
    expect(questionBox().value).toBe('What are the symptoms of asthma?')
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('runs the same search again when Retrieve is pressed again', async () => {
    const fetchMock = respondWith(response())
    renderAt('/retrieval', 'BGE')

    ask('What are the symptoms of asthma?')
    await screen.findByRole('heading', { name: 'Retrieved chunks' })
    fireEvent.click(retrieveButton())

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
  })
})
