import { useEffect, useState } from 'react'
import { fetchDatasetSummary, type DatasetSummary } from '../api'
import { TOTAL_CHUNKS } from '../data/articleCatalogue'
import { EMBEDDING_MODELS } from '../data/embeddingModels'
import { ARTICLES } from '../data/knowledgeBase'

type RequestState =
  | { kind: 'loading' }
  | { kind: 'loaded'; summary: DatasetSummary }
  | { kind: 'failed'; message: string }

// The verified figures the frontend already holds (PRD 1.3 to 1.6).
const VERIFIED = {
  articles: ARTICLES.length,
  chunks: TOTAL_CHUNKS,
  embeddingIndexes: EMBEDDING_MODELS.map((model) => model.id).sort(),
}

const START_COMMAND = 'cd backend\nuv run uvicorn medquery_api.main:app --reload --host 127.0.0.1 --port 8000'

const count = (value: number) => value.toLocaleString('en-GB')

function differencesFrom(summary: DatasetSummary): string[] {
  const differences: string[] = []
  if (summary.articles !== VERIFIED.articles) {
    differences.push(`Articles: ${count(summary.articles)} (verified: ${count(VERIFIED.articles)})`)
  }
  if (summary.chunks !== VERIFIED.chunks) {
    differences.push(`Chunks: ${count(summary.chunks)} (verified: ${count(VERIFIED.chunks)})`)
  }
  const indexes = [...summary.embeddingIndexes].sort().join(', ')
  if (indexes !== VERIFIED.embeddingIndexes.join(', ')) {
    differences.push(`Embedding indexes: ${indexes || 'none'} (verified: ${VERIFIED.embeddingIndexes.join(', ')})`)
  }
  return differences
}

// Asks the backend to count the research dataset and shows the answer next to the
// verified figures. The verified figures themselves are never replaced.
function BackendStatus() {
  const [state, setState] = useState<RequestState>({ kind: 'loading' })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    // AbortController cancels the request if the panel disappears before it finishes.
    const controller = new AbortController()
    fetchDatasetSummary(controller.signal)
      .then((summary) => setState({ kind: 'loaded', summary }))
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setState({ kind: 'failed', message: error instanceof Error ? error.message : String(error) })
        }
      })
    return () => controller.abort()
  }, [attempt])

  function tryAgain() {
    setState({ kind: 'loading' })
    setAttempt((n) => n + 1)
  }

  return (
    <div className="config-panel backend-panel" aria-live="polite">
      {state.kind === 'loading' && <p className="status-line">Checking the backend…</p>}

      {state.kind === 'failed' && (
        <>
          <p className="status-line">
            <span aria-hidden="true">○ </span>Backend not available
          </p>
          <p>{state.message}</p>
          <p>Start it in a second terminal, then try again:</p>
          <pre className="command">{START_COMMAND}</pre>
          <button type="button" className="button-secondary" onClick={tryAgain}>
            Try again
          </button>
        </>
      )}

      {state.kind === 'loaded' && (
        <>
          <p className="status-line">
            <span aria-hidden="true">● </span>Connected to the MedQuery backend
          </p>
          <p>
            Counted live by the backend from <code>{state.summary.source}</code> (read-only):
          </p>
          <p className="live-counts">
            <strong>{count(state.summary.articles)}</strong> articles ·{' '}
            <strong>{count(state.summary.chunks)}</strong> chunks ·{' '}
            <strong>{state.summary.embeddingIndexes.length}</strong> embedding indexes (
            {state.summary.embeddingIndexes.join(', ')})
          </p>
          {differencesFrom(state.summary).length === 0 ? (
            <p className="check-ok">
              <span aria-hidden="true">✓ </span>Matches the verified dataset figures above.
            </p>
          ) : (
            <div className="check-warning" role="alert">
              <p>
                <span aria-hidden="true">⚠ </span>The backend&apos;s counts differ from the verified
                figures. The verified figures above are unchanged.
              </p>
              <ul className="plain-list">
                {differencesFrom(state.summary).map((difference) => (
                  <li key={difference}>{difference}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default BackendStatus
