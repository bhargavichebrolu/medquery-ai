// The search on the Retrieval page, as kept in the address: ?q=question&model=BGE&k=5.
// The limits mirror POST /api/retrieve; the backend still checks every request itself.
import { isModelId, type ModelId } from './data/embeddingModels'

export const QUESTION_MAX_CHARACTERS = 1000
export const TOP_K_MIN = 1
export const TOP_K_MAX = 20
export const TOP_K_DEFAULT = 5

export interface Search {
  question: string // already trimmed
  model: ModelId
  topK: number
}

// Characters as Python counts them (code points), after removing surrounding whitespace.
export function questionLength(question: string): number {
  return [...question.trim()].length
}

export function questionProblem(question: string): string | null {
  const length = questionLength(question)
  if (length === 0) {
    return 'Enter a question.'
  }
  if (length > QUESTION_MAX_CHARACTERS) {
    return `Shorten the question to ${QUESTION_MAX_CHARACTERS.toLocaleString('en-GB')} characters or fewer.`
  }
  return null
}

export function parseTopK(value: string | null): number | null {
  if (value === null || !/^\d{1,2}$/.test(value)) {
    return null
  }
  const topK = Number(value)
  return topK >= TOP_K_MIN && topK <= TOP_K_MAX ? topK : null
}

// What the address holds. Each part is kept only if it is valid; `search` is set only
// when all three are, so an incomplete or invalid address never starts a request.
export interface AddressSearch {
  question: string | null
  model: ModelId | null
  topK: number | null
  search: Search | null
}

export function parseAddress(params: URLSearchParams): AddressSearch {
  const question = params.get('q')
  const model = params.get('model')
  const validModel = isModelId(model) ? model : null
  const topK = parseTopK(params.get('k'))
  const search =
    question !== null && questionProblem(question) === null && validModel && topK
      ? { question: question.trim(), model: validModel, topK }
      : null
  return { question, model: validModel, topK, search }
}

export function toAddress(search: Search): URLSearchParams {
  return new URLSearchParams({ q: search.question, model: search.model, k: String(search.topK) })
}
