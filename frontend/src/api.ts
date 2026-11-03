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

export function asApiError(error: unknown, message: string): ApiError {
  // Unknown errors are never shown as they are: they could contain internal details.
  return error instanceof ApiError ? error : new ApiError('unexpected', message)
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

// Send a request to the backend and return its JSON body. Throws an ApiError for every kind
// of failure; an aborted request rejects with the browser's AbortError.
async function requestJson(
  url: string,
  init: RequestInit,
  unavailableMessage: string,
): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(url, init)
  } catch (error) {
    if (init.signal?.aborted) throw error
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
    throw new ApiError('unavailable', typeof detail === 'string' ? detail : unavailableMessage)
  }
  if (!response.ok) {
    throw new ApiError('unexpected', `The backend answered with HTTP ${response.status}.`)
  }
  return readJson(response)
}

const postJson = (body: unknown, signal?: AbortSignal): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
  signal,
})

const UNEXPECTED_FORMAT = 'The backend sent a response in an unexpected format.'

// Ask the backend for the chunks most similar to a question.
export async function retrieveChunks(
  request: RetrievalRequest,
  signal?: AbortSignal,
): Promise<RetrievalResponse> {
  const body = await requestJson(
    '/api/retrieve',
    postJson(request, signal),
    'The backend cannot retrieve chunks right now.',
  )
  if (!isRetrievalResponse(body)) {
    throw new ApiError('unexpected', UNEXPECTED_FORMAT)
  }
  return body
}

// ---------- The configured LLM (GET /api/llm) ----------

export interface LlmConfig {
  provider: string // "mock" or "claude"
  model: string
}

function isLlmConfig(value: unknown): value is LlmConfig {
  const v = value as LlmConfig
  return typeof v === 'object' && v !== null && typeof v.provider === 'string' && typeof v.model === 'string'
}

// Which LLM the backend answers with. Credentials are never sent to the frontend.
export async function fetchLlmConfig(signal?: AbortSignal): Promise<LlmConfig> {
  const body = await requestJson(
    '/api/llm',
    { signal },
    'The backend cannot say which language model it uses right now.',
  )
  if (!isLlmConfig(body)) {
    throw new ApiError('unexpected', UNEXPECTED_FORMAT)
  }
  return body
}

// ---------- Grounded answers (POST /api/answer) ----------

// A citation that passed the backend's checks: citedText is exactly that span of the source.
export interface AnswerCitation {
  documentIndex: number
  sourceLabel: string // "S1"...; matches AnswerSource.label
  citedText: string
  startChar: number // positions count characters (code points), as Python does
  endChar: number
}

export interface AnswerPart {
  text: string
  citations: AnswerCitation[]
}

// How far valid citations support an answer (rules in backend/src/medquery_api/llm/citations.py).
export const GROUNDINGS = ['grounded', 'partly_grounded', 'not_grounded', 'not_enough_evidence', 'refused'] as const
export type Grounding = (typeof GROUNDINGS)[number]

// One sentence of the answer: positions count characters (code points) in all parts' text joined.
export interface AnswerSentence {
  start: number
  end: number
  supported: boolean // at least one valid citation supports it
}

export interface AnswerSource {
  label: string
  rank: number
  score: number
  sourceFile: string
  chunkId: number
  articleTitle: string
  chunkCount: number
  title: string
  text: string // verbatim chunk text
}

export interface CitationCheck {
  total: number
  valid: number
  dropped: { unknownDocument: number; invalidRange: number; textMismatch: number }
  validityRate: number | null // null when the answer had no citations
  sentences: number
  supportedSentences: number
  groundingCoverage: number | null // null when the answer had no sentences
}

export interface AnswerResponse {
  question: string
  embeddingModel: ModelId
  topK: number
  llm: { provider: string; model: string; fellBack: boolean }
  parts: AnswerPart[]
  grounding: Grounding
  sentences: AnswerSentence[]
  citationStatus: 'checked'
  citationCheck: CitationCheck
  droppedCitations: unknown[]
  sources: AnswerSource[]
  stopReason: string
  refused: boolean
  refusalCategory: string | null
  usage: {
    inputTokens: number
    outputTokens: number
    cacheReadInputTokens: number
    cacheCreationInputTokens: number
  }
  cost: { estimatedUsd: number | null; pricesCheckedOn: string | null; note: string }
  timings: {
    modelLoadMs: number
    embeddingMs: number
    searchMs: number
    generationMs: number
    totalMs: number
  }
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null
const isRate = (value: unknown) => value === null || isNumber(value)
const allWholeNumbers = (value: unknown, keys: string[]) =>
  isObject(value) && keys.every((key) => isWholeNumber(value[key]))

function isAnswerCitation(value: unknown): value is AnswerCitation {
  return (
    isObject(value) &&
    isWholeNumber(value.documentIndex) &&
    typeof value.sourceLabel === 'string' &&
    typeof value.citedText === 'string' &&
    isWholeNumber(value.startChar) &&
    isWholeNumber(value.endChar)
  )
}

function isAnswerPart(value: unknown): value is AnswerPart {
  return (
    isObject(value) &&
    typeof value.text === 'string' &&
    Array.isArray(value.citations) &&
    value.citations.every(isAnswerCitation)
  )
}

function isAnswerSource(value: unknown): value is AnswerSource {
  return (
    isObject(value) &&
    typeof value.label === 'string' &&
    isWholeNumber(value.rank) &&
    isNumber(value.score) &&
    typeof value.sourceFile === 'string' &&
    isWholeNumber(value.chunkId) &&
    typeof value.articleTitle === 'string' &&
    isWholeNumber(value.chunkCount) &&
    typeof value.title === 'string' &&
    typeof value.text === 'string'
  )
}

function isAnswerResponse(value: unknown): value is AnswerResponse {
  if (!isObject(value)) return false
  const { llm, citationCheck: check, usage, cost, timings } = value
  return (
    typeof value.question === 'string' &&
    isModelId(value.embeddingModel) &&
    isWholeNumber(value.topK) &&
    isObject(llm) &&
    typeof llm.provider === 'string' &&
    typeof llm.model === 'string' &&
    typeof llm.fellBack === 'boolean' &&
    Array.isArray(value.parts) &&
    value.parts.every(isAnswerPart) &&
    GROUNDINGS.includes(value.grounding as Grounding) &&
    Array.isArray(value.sentences) &&
    value.sentences.every(
      (s) => allWholeNumbers(s, ['start', 'end']) && typeof (s as AnswerSentence).supported === 'boolean',
    ) &&
    value.citationStatus === 'checked' &&
    allWholeNumbers(check, ['total', 'valid', 'sentences', 'supportedSentences']) &&
    isObject(check) &&
    allWholeNumbers(check.dropped, ['unknownDocument', 'invalidRange', 'textMismatch']) &&
    isRate(check.validityRate) &&
    isRate(check.groundingCoverage) &&
    Array.isArray(value.droppedCitations) &&
    Array.isArray(value.sources) &&
    value.sources.every(isAnswerSource) &&
    typeof value.stopReason === 'string' &&
    typeof value.refused === 'boolean' &&
    (value.refusalCategory === null || typeof value.refusalCategory === 'string') &&
    allWholeNumbers(usage, ['inputTokens', 'outputTokens', 'cacheReadInputTokens', 'cacheCreationInputTokens']) &&
    isObject(cost) &&
    isRate(cost.estimatedUsd) &&
    (cost.pricesCheckedOn === null || typeof cost.pricesCheckedOn === 'string') &&
    typeof cost.note === 'string' &&
    isObject(timings) &&
    ['modelLoadMs', 'embeddingMs', 'searchMs', 'generationMs', 'totalMs'].every((key) =>
      isNumber(timings[key]),
    )
  )
}

// Ask the configured LLM to answer one question from the chunks retrieved for it. Each call
// is independent: no earlier questions or answers are sent.
export async function askQuestion(
  request: RetrievalRequest,
  signal?: AbortSignal,
): Promise<AnswerResponse> {
  const body = await requestJson(
    '/api/answer',
    postJson(request, signal),
    'The backend cannot answer questions right now.',
  )
  if (!isAnswerResponse(body)) {
    throw new ApiError('unexpected', UNEXPECTED_FORMAT)
  }
  return body
}

// ---------- Evaluation dataset (GET /api/evaluation/dataset, PRD 4.0) ----------

// Historical text from the research, not ground truth: no reference answer has been checked.
export interface ReferenceAnswer {
  text: string
  status: 'unverified'
  source: string
}

// A question the knowledge base can answer, with the articles that are relevant to it.
export interface InScopeQuestion {
  id: string
  question: string
  relevantArticles: string[] // article file names, such as "asthma.txt"
  referenceAnswer: ReferenceAnswer | null
  archiveQuestionId: number | null // the research's question number, when it has one
  source: string
}

// A question that should not get an ordinary answer, with its approved acceptance criteria.
export interface OutOfScopeQuestion {
  id: string
  question: string
  reason: string
  acceptanceCriteria: string
  automaticPassStatuses: Grounding[] // outcomes that pass without review
  otherwise: 'fail' | 'manual_review' // what any other outcome means
  source: string
}

export interface EvaluationDataset {
  identifier: string // "sha256:" and a hash of the content: a new identifier for any change
  formatVersion: number
  name: string
  description: string
  relevanceLevel: 'article'
  inScopeQuestions: InScopeQuestion[]
  outOfScopeQuestions: OutOfScopeQuestion[]
}

const isStringList = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string')

function isInScopeQuestion(value: unknown): value is InScopeQuestion {
  if (!isObject(value)) return false
  const reference = value.referenceAnswer
  return (
    typeof value.id === 'string' &&
    typeof value.question === 'string' &&
    isStringList(value.relevantArticles) &&
    (reference === null ||
      (isObject(reference) &&
        typeof reference.text === 'string' &&
        reference.status === 'unverified' &&
        typeof reference.source === 'string')) &&
    (value.archiveQuestionId === null || isWholeNumber(value.archiveQuestionId)) &&
    typeof value.source === 'string'
  )
}

function isOutOfScopeQuestion(value: unknown): value is OutOfScopeQuestion {
  return (
    isObject(value) &&
    typeof value.id === 'string' &&
    typeof value.question === 'string' &&
    typeof value.reason === 'string' &&
    typeof value.acceptanceCriteria === 'string' &&
    Array.isArray(value.automaticPassStatuses) &&
    value.automaticPassStatuses.every((status) => (GROUNDINGS as readonly unknown[]).includes(status)) &&
    (value.otherwise === 'fail' || value.otherwise === 'manual_review') &&
    typeof value.source === 'string'
  )
}

function isEvaluationDataset(value: unknown): value is EvaluationDataset {
  return (
    isObject(value) &&
    typeof value.identifier === 'string' &&
    isWholeNumber(value.formatVersion) &&
    typeof value.name === 'string' &&
    typeof value.description === 'string' &&
    value.relevanceLevel === 'article' &&
    Array.isArray(value.inScopeQuestions) &&
    value.inScopeQuestions.every(isInScopeQuestion) &&
    Array.isArray(value.outOfScopeQuestions) &&
    value.outOfScopeQuestions.every(isOutOfScopeQuestion)
  )
}

// The questions Part 4 evaluates with. Read-only: nothing is run or scored.
export async function fetchEvaluationDataset(signal?: AbortSignal): Promise<EvaluationDataset> {
  const body = await requestJson(
    '/api/evaluation/dataset',
    { signal },
    'The backend cannot show the evaluation dataset right now.',
  )
  if (!isEvaluationDataset(body)) {
    throw new ApiError('unexpected', UNEXPECTED_FORMAT)
  }
  return body
}
