// Requests to the MedQuery backend. Addresses start with /api; in development the
// Vite server forwards them to the Python backend (see vite.config.ts).

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
