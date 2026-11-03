import { START_COMMAND, type ApiError } from '../api'

// A failed backend request: what went wrong, how to start the backend when it is not running,
// and a way to try again.
function ErrorPanel({ title, error, onRetry }: { title: string; error: ApiError; onRetry?: () => void }) {
  return (
    <div className="config-panel retrieval-status" role="alert">
      <p className="status-line">{error.kind === 'unreachable' ? 'Backend not available' : title}</p>
      <p>{error.message}</p>
      {error.kind === 'unreachable' && (
        <>
          <p>Start it in a second terminal, then try again:</p>
          <pre className="command">{START_COMMAND}</pre>
        </>
      )}
      {onRetry && (
        <button type="button" className="button-secondary" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  )
}

export default ErrorPanel
