// Requests to the MedQuery backend. Addresses start with /api; in development the
// Vite server forwards them to the Python backend (see vite.config.ts).
import { isModelId, type ModelId } from './data/embeddingModels'

// How to start the backend, shown whenever it cannot be reached.
export const START_COMMAND =
  'cd backend\nuv run uvicorn medquery_api.main:app --reload --host 127.0.0.1 --port 8000'

export interface DatasetSummary {
  articles: number
  chunks: number
  embeddingIndexes: string[]
  source: string
}

function isDatasetSummary(value: unknown): value is DatasetSummary {
  const v = value as DatasetSummary
  return (
    typeof v === 'object' &&
    v !== null &&
    Number.isInteger(v.articles) &&
    Number.isInteger(v.chunks) &&
    Array.isArray(v.embeddingIndexes) &&
    v.embeddingIndexes.every((name) => typeof name === 'string') &&
    typeof v.source === 'string'
  )
}

// Ask the backend to count the research dataset. Throws if the backend cannot be
// reached, answers with an error, or sends something unexpected.
export async function fetchDatasetSummary(signal?: AbortSignal): Promise<DatasetSummary> {
  const response = await fetch('/api/dataset/summary', { signal })
  if (response.status === 502 || response.status === 504) {
    // The development proxy answers 502/504 itself when nothing is listening on the backend port.
    throw new Error(`Could not reach the backend (HTTP ${response.status} from the development proxy).`)
  }
  if (!response.ok) {
    // FastAPI puts its error message in "detail"; include it when there is one.
    const detail = await response
      .json()
      .then((body: { detail?: unknown }) => (typeof body.detail === 'string' ? ` ${body.detail}` : ''))
      .catch(() => '')
    throw new Error(`The backend answered with HTTP ${response.status}.${detail}`)
  }
  const body: unknown = await response.json()
  if (!isDatasetSummary(body)) {
    throw new Error('The backend sent a response in an unexpected format.')
  }
  return body
}

// ---------- Retrieval (POST /api/retrieve) ----------

export interface RetrievalRequest {
  question: string
  model: ModelId
  topK: number
}

export interface RetrievedChunk {
  rank: number
  score: number // cosine similarity between the question and the chunk
  sourceFile: string
  chunkId: number
  chunkLength: number
  text: string
}

export interface RetrievalResponse {
  question: string
  model: ModelId
  modelId: string
  revision: string
  topK: number
  results: RetrievedChunk[]
  timings: { modelLoadMs: number; embeddingMs: number; searchMs: number }
}

// Why a request failed, so the page can show the right help.
export type ApiErrorKind = 'unreachable' | 'invalid' | 'unavailable' | 'unexpected'

export class ApiError extends Error {
  readonly kind: ApiErrorKind

  constructor(kind: ApiErrorKind, message: string) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
  }
}

const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const isWholeNumber = (value: unknown): value is number => Number.isInteger(value)

function isRetrievedChunk(value: unknown): value is RetrievedChunk {
  const v = value as RetrievedChunk
  return (
    typeof v === 'object' &&
    v !== null &&
    isWholeNumber(v.rank) &&
    isNumber(v.score) &&
    typeof v.sourceFile === 'string' &&
    isWholeNumber(v.chunkId) &&
    isWholeNumber(v.chunkLength) &&
    typeof v.text === 'string'
  )
}

function isRetrievalResponse(value: unknown): value is RetrievalResponse {
  const v = value as RetrievalResponse
  return (
    typeof v === 'object' &&
    v !== null &&
    typeof v.question === 'string' &&
    isModelId(v.model) &&
    typeof v.modelId === 'string' &&
    typeof v.revision === 'string' &&
    isWholeNumber(v.topK) &&
    Array.isArray(v.results) &&
    v.results.every(isRetrievedChunk) &&
    typeof v.timings === 'object' &&
    v.timings !== null &&
    isNumber(v.timings.modelLoadMs) &&
    isNumber(v.timings.embeddingMs) &&
    isNumber(v.timings.searchMs)
  )
}

// FastAPI explains a 422 as a list of problems, each naming the field ("loc") and the
// reason ("msg"). Turn that into one readable sentence.
const FIELD_NAMES: Record<string, string> = { question: 'Question', model: 'Model', topK: 'Number of results' }

function describeInvalidRequest(body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (!Array.isArray(detail)) {
    return 'The backend rejected the request as invalid.'
  }
  const problems = detail.flatMap((item: { loc?: unknown; msg?: unknown }) => {
    if (typeof item?.msg !== 'string') return []
    const field = Array.isArray(item.loc) ? String(item.loc[item.loc.length - 1]) : ''
    return [`${FIELD_NAMES[field] ?? field}: ${item.msg}`.replace(/^: /, '')]
  })
  return problems.length > 0
    ? `The backend rejected the request: ${problems.join('; ')}.`
    : 'The backend rejected the request as invalid.'
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return null
  }
}

// Ask the backend for the chunks most similar to a question. Throws an ApiError for
// every kind of failure; an aborted request rejects with the browser's AbortError.
export async function retrieveChunks(
  request: RetrievalRequest,
  signal?: AbortSignal,
): Promise<RetrievalResponse> {
  let response: Response
  try {
    response = await fetch('/api/retrieve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new ApiError('unreachable', 'Could not reach the backend.')
  }

  if (response.status === 502 || response.status === 504) {
    // The development proxy answers 502/504 itself when nothing is listening on the backend port.
    throw new ApiError(
      'unreachable',
      `Could not reach the backend (HTTP ${response.status} from the development proxy).`,
    )
  }
  if (response.status === 422) {
    throw new ApiError('invalid', describeInvalidRequest(await readJson(response)))
  }
  if (response.status === 503) {
    // The backend's 503 messages are written for people and contain no file paths.
    const detail = (await readJson(response) as { detail?: unknown } | null)?.detail
    throw new ApiError(
      'unavailable',
      typeof detail === 'string' ? detail : 'The backend cannot retrieve chunks right now.',
    )
  }
  if (!response.ok) {
    throw new ApiError('unexpected', `The backend answered with HTTP ${response.status}.`)
  }
  const body = await readJson(response)
  if (!isRetrievalResponse(body)) {
    throw new ApiError('unexpected', 'The backend sent a response in an unexpected format.')
  }
  return body
}
