import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { describe, expect, it, vi } from 'vitest'
import App from '../App'
import type { AnswerResponse, LlmConfig } from '../api'
import { STORAGE_KEY } from '../state/modelSelection'
import { PRIVACY_NOTICE } from './ChatPage'

function Address() {
  const location = useLocation()
  return <span data-testid="address">{location.pathname + location.search + location.hash}</span>
}

function renderChat(savedModel: string | null = 'BGE') {
  if (savedModel) {
    localStorage.setItem(STORAGE_KEY, savedModel)
  }
  render(
    <MemoryRouter initialEntries={['/chat']}>
      <App />
      <Address />
    </MemoryRouter>,
  )
}

const questionBox = () => screen.getByRole('textbox', { name: 'Question' }) as HTMLTextAreaElement
const askButton = () => screen.getByRole('button', { name: 'Ask' })
const status = () => screen.getByRole('status')

function ask(question: string, topK?: number) {
  fireEvent.change(questionBox(), { target: { value: question } })
  if (topK) {
    fireEvent.change(screen.getByRole('combobox', { name: 'Number of chunks' }), {
      target: { value: String(topK) },
    })
  }
  fireEvent.click(askButton())
}

const ASTHMA_TEXT = '# Asthma\n\nAsthma is a common lung condition.\n\nSee https://www.example.org/help.'
const HIVES_TEXT = 'Hives is a raised, itchy rash.'

function answer(overrides: Partial<AnswerResponse> = {}): AnswerResponse {
  const quoted = 'Asthma is a common lung condition.'
  const start = ASTHMA_TEXT.indexOf(quoted)
  return {
    question: 'What is asthma?',
    embeddingModel: 'BGE',
    topK: 2,
    llm: { provider: 'mock', model: 'mock', fellBack: false },
    parts: [
      { text: 'Asthma is ', citations: [] },
      {
        text: 'a common lung condition.',
        citations: [
          { documentIndex: 0, sourceLabel: 'S1', citedText: quoted, startChar: start, endChar: start + quoted.length },
        ],
      },
      { text: ' It is not a rash.', citations: [] },
    ],
    grounding: 'partly_grounded',
    sentences: [
      { start: 0, end: 34, supported: true }, // "Asthma is a common lung condition."
      { start: 35, end: 52, supported: false }, // "It is not a rash."
    ],
    citationStatus: 'checked',
    citationCheck: {
      total: 3,
      valid: 1,
      dropped: { unknownDocument: 1, invalidRange: 0, textMismatch: 1 },
      validityRate: 1 / 3,
      sentences: 2,
      supportedSentences: 1,
      groundingCoverage: 0.5,
    },
    droppedCitations: [{}, {}],
    sources: [
      { label: 'S1', rank: 1, score: 0.8265177011489868, sourceFile: 'asthma.txt', chunkId: 1, articleTitle: 'Asthma', chunkCount: 14, title: 'Asthma (chunk 1 of 14)', text: ASTHMA_TEXT },
      { label: 'S2', rank: 2, score: 0.61, sourceFile: 'not_in_catalogue.txt', chunkId: 3, articleTitle: 'Hives', chunkCount: 5, title: 'Hives (chunk 3 of 5)', text: HIVES_TEXT },
    ],
    stopReason: 'end_turn',
    refused: false,
    refusalCategory: null,
    usage: { inputTokens: 1234, outputTokens: 56, cacheReadInputTokens: 0, cacheCreationInputTokens: 0 },
    cost: { estimatedUsd: 0, pricesCheckedOn: null, note: 'The mock provider costs nothing.' },
    timings: { modelLoadMs: 2699.4, embeddingMs: 8.6, searchMs: 0.58, generationMs: 1520, totalMs: 4300 },
    ...overrides,
  }
}

type Reply = { body: unknown; status?: number } | 'network-error'

// A backend stand-in: GET /api/llm answers with the configuration, and each POST /api/answer
// takes the next reply in order. Every request is recorded.
function backend(config: LlmConfig | Reply = { provider: 'mock', model: 'mock' }, ...answers: Reply[]) {
  const requests: { url: string; method: string; body: unknown }[] = []
  const respond = (reply: Reply) =>
    reply === 'network-error'
      ? Promise.reject(new TypeError('Failed to fetch'))
      : Promise.resolve(new Response(JSON.stringify(reply.body), { status: reply.status ?? 200 }))
  const fetchMock = vi.fn((url: string, init: RequestInit = {}) => {
    requests.push({ url, method: init.method ?? 'GET', body: init.body ? JSON.parse(String(init.body)) : null })
    if (url === '/api/llm') {
      return respond(config === 'network-error' || 'body' in config ? config : { body: config })
    }
    return respond(answers.shift() ?? { body: answer() })
  })
  vi.stubGlobal('fetch', fetchMock)
  return requests
}

// Wait until the page knows which language model the backend uses.
async function ready() {
  await waitFor(() =>
    expect(screen.queryByText('Checking which language model the backend uses…')).not.toBeInTheDocument(),
  )
}

describe('Chat: the configured language model', () => {
  it('waits for the backend to say which model it uses before allowing a question', async () => {
    backend()
    renderChat()
    fireEvent.change(questionBox(), { target: { value: 'What is asthma?' } })

    expect(askButton()).toBeDisabled()
    expect(screen.getByText('Checking which language model the backend uses…')).toBeInTheDocument()
    await ready()
  })

  it('shows the approved privacy notice when Claude answers', async () => {
    backend({ provider: 'claude', model: 'claude-sonnet-5-5' })
    renderChat()

    const notice = await screen.findByRole('region', { name: 'Privacy' })
    expect(notice).toHaveTextContent(PRIVACY_NOTICE)
    expect(PRIVACY_NOTICE).toBe(
      'Answers are generated by Claude, a model hosted by Anthropic. Your question and retrieved NHS text are sent to ' +
        "Anthropic's API to generate each answer. Avoid entering personal or identifying health information. This " +
        'tool provides general information, not medical advice. In an emergency, call 999.',
    )
  })

  it('explains the mock provider and shows no privacy notice for it', async () => {
    backend()
    renderChat()

    expect(await screen.findByText(/The backend is using the mock provider/)).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Privacy' })).not.toBeInTheDocument()
  })

  it('shows the notice if an answer comes from Claude even when the mock was configured', async () => {
    backend({ provider: 'mock', model: 'mock' }, { body: answer({ llm: { provider: 'claude', model: 'claude-sonnet-5-5', fellBack: false } }) })
    renderChat()
    await ready()

    ask('What is asthma?')

    expect(await screen.findByRole('region', { name: 'Privacy' })).toHaveTextContent(PRIVACY_NOTICE)
  })

  it('offers the start command and a retry when the backend cannot be reached', async () => {
    const requests = backend('network-error')
    renderChat()

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Backend not available')
    expect(alert).toHaveTextContent('uv run uvicorn medquery_api.main:app')
    expect(askButton()).toBeDisabled()
    expect(screen.getByText('Waiting for the backend.')).toBeInTheDocument()

    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(JSON.stringify({ provider: 'mock', model: 'mock' })))))
    fireEvent.click(within(alert).getByRole('button', { name: 'Try again' }))
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
    expect(requests).toHaveLength(1)
  })

  it('shows a misconfigured provider as a safe message', async () => {
    backend({ status: 503, body: { detail: 'The LLM provider is not configured correctly.' } })
    renderChat()

    expect(await screen.findByRole('alert')).toHaveTextContent('The LLM provider is not configured correctly.')
  })
})

describe('Chat: asking', () => {
  it('needs a question and an embedding model', async () => {
    backend()
    renderChat(null)
    await waitFor(() => expect(screen.getByText('Choose an embedding model.')).toBeInTheDocument())

    fireEvent.click(screen.getByRole('radio', { name: 'MPNet' }))
    expect(screen.getByText('Enter a question.')).toBeInTheDocument()
    fireEvent.change(questionBox(), { target: { value: 'x'.repeat(1001) } })
    expect(screen.getByText('Shorten the question to 1,000 characters or fewer.')).toBeInTheDocument()
    expect(askButton()).toBeDisabled()
  })

  it('sends only the trimmed question, the model and top-k, and keeps the address unchanged', async () => {
    const requests = backend()
    renderChat()
    await ready()

    ask('  What is asthma?  ', 2)
    await screen.findByRole('heading', { name: 'Answer' })

    expect(requests.at(-1)).toEqual({
      url: '/api/answer',
      method: 'POST',
      body: { question: 'What is asthma?', model: 'BGE', topK: 2 },
    })
    expect(screen.getByTestId('address')).toHaveTextContent(/^\/chat$/)
  })

  it('treats each question independently: nothing earlier is sent, and the new answer replaces the old', async () => {
    const requests = backend(
      { provider: 'mock', model: 'mock' },
      { body: answer() },
      { body: answer({ question: 'What is hives?', parts: [{ text: 'Hives is a rash.', citations: [] }] }) },
    )
    renderChat()
    await ready()

    ask('What is asthma?')
    await screen.findByText('a common lung condition.')
    ask('What is hives?')
    await screen.findByText('Hives is a rash.')

    expect(requests.at(-1)?.body).toEqual({ question: 'What is hives?', model: 'BGE', topK: 5 })
    expect(screen.queryByText('a common lung condition.')).not.toBeInTheDocument()
    expect(screen.getAllByRole('heading', { name: 'Answer' })).toHaveLength(1)
  })

  it('shows that it is answering and blocks a second question meanwhile', async () => {
    backend()
    let release: (response: Response) => void = () => {}
    renderChat()
    await ready()
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>((resolve) => (release = resolve))))

    ask('What is asthma?')

    expect(status()).toHaveTextContent('Retrieving chunks and generating an answer…')
    expect(screen.getByText('Answering…')).toBeInTheDocument()
    expect(askButton()).toBeDisabled()
    await act(async () => release(new Response(JSON.stringify(answer()))))
    expect(status()).toHaveTextContent('Answer ready: partly supported by the retrieved sources.')
    expect(askButton()).toBeEnabled()
  })

  it('explains a rejected question without offering a retry', async () => {
    backend(
      { provider: 'mock', model: 'mock' },
      { status: 422, body: { detail: [{ loc: ['body', 'topK'], msg: 'Input should be less than or equal to 20' }] } },
    )
    renderChat()
    await ready()

    ask('What is asthma?')

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('The question was not accepted')
    expect(alert).toHaveTextContent('Number of results: Input should be less than or equal to 20')
    expect(within(alert).queryByRole('button')).not.toBeInTheDocument()
  })

  it('shows a backend failure and asks the same question again on retry', async () => {
    const requests = backend(
      { provider: 'mock', model: 'mock' },
      { status: 503, body: { detail: "Claude's rate limit was reached. Try again shortly." } },
      { body: answer() },
    )
    renderChat()
    await ready()

    ask('What is asthma?')
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent("Claude's rate limit was reached. Try again shortly.")
    fireEvent.click(within(alert).getByRole('button', { name: 'Try again' }))

    await screen.findByRole('heading', { name: 'Answer' })
    expect(requests.filter((r) => r.url === '/api/answer').map((r) => r.body)).toEqual([
      { question: 'What is asthma?', model: 'BGE', topK: 5 },
      { question: 'What is asthma?', model: 'BGE', topK: 5 },
    ])
  })

  it('treats a malformed response as an error', async () => {
    backend({ provider: 'mock', model: 'mock' }, { body: { ...answer(), citationStatus: 'unchecked' } })
    renderChat()
    await ready()

    ask('What is asthma?')

    expect(await screen.findByRole('alert')).toHaveTextContent('The backend sent a response in an unexpected format.')
  })
})

describe('Chat: the answer', () => {
  async function showAnswer(response = answer()) {
    backend({ provider: 'mock', model: 'mock' }, { body: response })
    renderChat()
    await ready()
    ask(response.question)
    return screen.findByRole('region', { name: 'Answer' })
  }

  it('shows the answer with a marker for each cited source, linked to its card', async () => {
    const section = await showAnswer()

    const marker = within(section).getByRole('link', { name: 'Source S1' })
    expect(marker.parentElement).toHaveTextContent('a common lung condition.S1')
    expect(within(section).getAllByRole('link', { name: /^Source / })).toHaveLength(1)

    fireEvent.click(marker)
    const card = document.getElementById(marker.getAttribute('href')!.slice(1))!
    expect(card).toHaveFocus()
    expect(card).toHaveTextContent('Asthma')
    expect(screen.getByTestId('address')).toHaveTextContent(/^\/chat$/)
  })

  it('shows each source verbatim with its title, rank, score and the quoted passage highlighted', async () => {
    const section = await showAnswer()
    const cards = within(section).getAllByRole('listitem')

    expect(cards).toHaveLength(2)
    const [first, second] = cards
    expect(within(first).getByRole('link', { name: 'Asthma' })).toHaveAttribute('href', '/knowledge-base/asthma')
    expect(first).toHaveTextContent('Rank 1 · Score 0.8265')
    expect(first).toHaveTextContent('asthma.txt · chunk 1 of 14')
    expect(first.querySelector('.chunk-text')!.textContent).toBe(ASTHMA_TEXT)
    expect(first.querySelector('mark')).toHaveTextContent('Asthma is a common lung condition.')
    expect(within(first).getByText('Cited')).toBeInTheDocument()
    expect(within(first).queryByRole('link', { name: /example\.org/ })).not.toBeInTheDocument()

    expect(second.querySelector('.chunk-text')!.textContent).toBe(HIVES_TEXT)
    expect(second.querySelector('mark')).toBeNull()
    expect(within(second).queryByText('Cited')).not.toBeInTheDocument()
    expect(within(second).queryByRole('link')).not.toBeInTheDocument() // not in the catalogue
  })

  it('reports the citation checks as checks, not proof', async () => {
    const section = await showAnswer()
    const checks = within(section).getByRole('heading', { name: 'Citation checks' }).parentElement!

    expect(checks).toHaveTextContent('Valid citations1 of 3 (33%)')
    expect(checks).toHaveTextContent(
      'Dropped citations2: 1 named no retrieved source, 1 quoted text that did not match its source',
    )
    expect(checks).toHaveTextContent('Grounding coverage1 of 2 sentences have a valid citation (50%)')
    expect(checks).toHaveTextContent('These are checks, not proof.')
  })

  it('says when there were no citations or sentences to check', async () => {
    const section = await showAnswer(
      answer({
        parts: [{ text: 'No citations here.', citations: [] }],
        grounding: 'not_grounded',
        sentences: [],
        citationCheck: {
          total: 0,
          valid: 0,
          dropped: { unknownDocument: 0, invalidRange: 0, textMismatch: 0 },
          validityRate: null,
          sentences: 0,
          supportedSentences: 0,
          groundingCoverage: null,
        },
      }),
    )

    expect(section).toHaveTextContent('The answer has no citations.')
    expect(section).toHaveTextContent('Dropped citations0')
    expect(section).toHaveTextContent('No sentences to check.')
  })

  it('shows the model, fallback, tokens, cost and timings', async () => {
    const section = await showAnswer(
      answer({
        llm: { provider: 'claude', model: 'claude-opus-4-8', fellBack: true },
        usage: { inputTokens: 1234, outputTokens: 56, cacheReadInputTokens: 10, cacheCreationInputTokens: 0 },
        cost: { estimatedUsd: null, pricesCheckedOn: null, note: 'No checked price for claude-opus-4-8, so no estimate is given.' },
      }),
    )
    const details = within(section).getByRole('heading', { name: 'How this answer was made' }).parentElement!

    expect(details).toHaveTextContent(
      'Claude (Anthropic) · claude-opus-4-8 · a fallback model answered after the requested model declined',
    )
    expect(details).toHaveTextContent('RetrievalBGE, top 2')
    expect(details).toHaveTextContent('1,234 input · 56 output · 10 cache read · 0 cache write')
    expect(details).toHaveTextContent('Not estimated. No checked price for claude-opus-4-8')
    expect(details).toHaveTextContent(
      'model load 2.7 s · embedding 8.6 ms · search 0.6 ms · generation 1.5 s · total 4.3 s',
    )
  })

  it('shows the cost of a mock answer as zero', async () => {
    const section = await showAnswer()

    expect(section).toHaveTextContent('$0.0000. The mock provider costs nothing.')
    expect(section).toHaveTextContent('Mock provider (no language model) · mock')
  })

  it('shows a refusal as a clear message and no answer text', async () => {
    const section = await showAnswer(
      answer({
        stopReason: 'refusal',
        refused: true,
        grounding: 'refused',
        refusalCategory: 'bio',
        parts: [{ text: 'Partial text before the refusal', citations: [] }],
      }),
    )

    expect(section).toHaveTextContent('The language model declined to answer this question (category: bio)')
    expect(section).not.toHaveTextContent('Partial text before the refusal')
  })

  it('says when an answer may be cut off', async () => {
    const section = await showAnswer(answer({ stopReason: 'max_tokens' }))

    expect(section).toHaveTextContent('The answer reached the length limit and may be cut off.')
  })
})

describe('Chat: grounding (PRD 3.6)', () => {
  async function showAnswer(response: AnswerResponse) {
    backend({ provider: 'mock', model: 'mock' }, { body: response })
    renderChat()
    await ready()
    ask(response.question)
    return screen.findByRole('region', { name: 'Answer' })
  }

  const unsupportedText = (section: HTMLElement) =>
    [...section.querySelectorAll('.unsupported')].map((element) => element.textContent)

  const cited = answer().parts[1].citations

  it('presents a fully cited answer as supported, with nothing marked', async () => {
    const section = await showAnswer(
      answer({
        parts: [{ text: 'Asthma is a common lung condition.', citations: cited }],
        grounding: 'grounded',
        sentences: [{ start: 0, end: 34, supported: true }],
      }),
    )

    expect(section).toHaveTextContent('Supported by the retrieved sources.')
    expect(section).toHaveTextContent('This is a check, not proof that the answer is correct.')
    expect(unsupportedText(section)).toEqual([])
    expect(within(section).queryByText('No valid citation')).not.toBeInTheDocument()
    expect(status()).toHaveTextContent('Answer ready: supported by the retrieved sources.')
  })

  it('marks each unsupported sentence of a partly supported answer, visibly and in text', async () => {
    const section = await showAnswer(answer())

    expect(section).toHaveTextContent('Partly supported. 1 of 2 sentences have no valid citation')
    expect(unsupportedText(section)).toEqual(['It is not a rash.'])
    const label = within(section).getByText('No valid citation')
    expect(label.previousElementSibling).toHaveTextContent('It is not a rash.')
    expect(section.querySelector('.answer-text')).toHaveTextContent(
      'Asthma is a common lung condition.S1 It is not a rash.No valid citation',
    )
  })

  it('warns that a not-grounded answer is not supported and marks every sentence', async () => {
    const section = await showAnswer(
      answer({
        parts: [{ text: 'Asthma is caused by cats. It is rare.', citations: [] }],
        grounding: 'not_grounded',
        sentences: [
          { start: 0, end: 25, supported: false },
          { start: 26, end: 37, supported: false },
        ],
      }),
    )

    const warning = within(section).getByRole('note', { name: 'Not supported by the retrieved sources' })
    expect(warning).toHaveTextContent('may come from the model’s general knowledge and may be wrong. Do not rely on it')
    expect(section).not.toHaveTextContent('Supported by the retrieved sources.')
    expect(unsupportedText(section)).toEqual(['Asthma is caused by cats.', 'It is rare.'])
    expect(within(section).getAllByText('No valid citation')).toHaveLength(2)
    expect(within(section).queryByRole('link', { name: /^Source / })).not.toBeInTheDocument()
    expect(status()).toHaveTextContent('Answer ready: not supported by the retrieved sources.')
  })

  it('shows "not enough evidence" as a plain message, not as an answer', async () => {
    const sentence = 'The provided documents do not contain enough information to answer this question.'
    const section = await showAnswer(
      answer({
        parts: [{ text: sentence, citations: [] }],
        grounding: 'not_enough_evidence',
        sentences: [{ start: 0, end: sentence.length, supported: false }],
      }),
    )

    expect(section).toHaveTextContent(
      'Not enough evidenceThe retrieved NHS text does not contain enough information to answer this question',
    )
    expect(section.querySelector('.answer-text')).toBeNull()
    expect(within(section).queryByText('No valid citation')).not.toBeInTheDocument()
    expect(within(section).queryByRole('heading', { name: 'Citation checks' })).not.toBeInTheDocument()
    expect(within(section).getByRole('heading', { name: 'Sources the model was given' })).toBeInTheDocument()
    expect(status()).toHaveTextContent('Answer ready: not enough evidence to answer.')
  })

  it('announces a refusal and shows no citation checks for it', async () => {
    const section = await showAnswer(
      answer({ parts: [], grounding: 'refused', refused: true, stopReason: 'refusal', sentences: [] }),
    )

    expect(section).toHaveTextContent('The language model declined to answer this question')
    expect(within(section).queryByRole('heading', { name: 'Citation checks' })).not.toBeInTheDocument()
    expect(status()).toHaveTextContent('Answer ready: the language model declined to answer.')
  })

  it('marks a sentence split across parts once, after its last part', async () => {
    const section = await showAnswer(
      answer({
        parts: [
          { text: 'Asthma is a common lung condition.', citations: cited },
          { text: ' Cats ', citations: [] },
          { text: 'cure it.', citations: [] },
        ],
        grounding: 'partly_grounded',
        sentences: [
          { start: 0, end: 34, supported: true },
          { start: 35, end: 48, supported: false },
        ],
      }),
    )

    expect(unsupportedText(section)).toEqual(['Cats ', 'cure it.'])
    expect(within(section).getAllByText('No valid citation')).toHaveLength(1)
    expect(within(section).getByText('No valid citation').previousElementSibling).toHaveTextContent('cure it.')
  })

  it('counts positions in characters, so text beyond the basic plane does not shift the marks', async () => {
    const section = await showAnswer(
      answer({
        parts: [
          { text: '🙂 Asthma is common.', citations: cited },
          { text: ' Cats cure it.', citations: [] },
        ],
        grounding: 'partly_grounded',
        sentences: [
          { start: 0, end: 19, supported: true },
          { start: 20, end: 33, supported: false },
        ],
      }),
    )

    expect(unsupportedText(section)).toEqual(['Cats cure it.'])
  })

  it('rejects a response without a grounding status', async () => {
    const withoutGrounding: Partial<AnswerResponse> = answer()
    delete withoutGrounding.grounding
    backend({ provider: 'mock', model: 'mock' }, { body: withoutGrounding })
    renderChat()
    await ready()

    ask('What is asthma?')

    expect(await screen.findByRole('alert')).toHaveTextContent('The backend sent a response in an unexpected format.')
  })
})
