import { useEffect, useId, useMemo, useRef, useState, type FormEvent } from 'react'
import { useSearchParams } from 'react-router'
import { ApiError, retrieveChunks, START_COMMAND, type RetrievalResponse } from '../api'
import ModelOption from '../components/ModelOption'
import RetrievalResults from '../components/RetrievalResults'
import { EMBEDDING_MODELS } from '../data/embeddingModels'
import { EVALUATION_QUESTIONS } from '../data/researchResults'
import { PAGES } from '../navigation'
import {
  parseAddress,
  QUESTION_MAX_CHARACTERS,
  questionLength,
  questionProblem,
  toAddress,
  TOP_K_DEFAULT,
  TOP_K_MAX,
  TOP_K_MIN,
} from '../retrievalSearch'
import { useSelectedModel } from '../state/modelSelection'

type Outcome =
  | { key: string; kind: 'loaded'; response: RetrievalResponse }
  | { key: string; kind: 'failed'; error: ApiError }

const TOP_K_CHOICES = Array.from({ length: TOP_K_MAX - TOP_K_MIN + 1 }, (_, i) => TOP_K_MIN + i)

function asApiError(error: unknown): ApiError {
  // Unknown errors are never shown as they are: they could contain internal details.
  return error instanceof ApiError
    ? error
    : new ApiError('unexpected', 'Something went wrong while retrieving chunks.')
}

// The search lives in the address (?q=...&model=...&k=...). Submitting the form adds a new
// address to the history, so Back and Forward move between searches, and a link or a
// refresh repeats one. Requests start only from a complete, valid address.
function RetrievalPage() {
  const [params, setParams] = useSearchParams()
  const address = useMemo(() => parseAddress(params), [params])
  const addressKey = params.toString()
  const { selectedModel, selectModel } = useSelectedModel()

  // The form starts from the address, and is refilled when the address changes.
  const [question, setQuestion] = useState(address.question ?? '')
  const [topK, setTopK] = useState(address.topK ?? TOP_K_DEFAULT)
  const [filledFrom, setFilledFrom] = useState(addressKey)
  if (filledFrom !== addressKey) {
    setFilledFrom(addressKey)
    setQuestion(address.question ?? '')
    setTopK(address.topK ?? TOP_K_DEFAULT)
  }

  // A model named in the address becomes the selected model everywhere in the app.
  useEffect(() => {
    if (address.model) {
      selectModel(address.model)
    }
  }, [addressKey, address.model, selectModel])

  // One request per search (and per "Try again"). A newer search or leaving the page aborts
  // the older request, and an aborted request never updates the page.
  const search = address.search
  const [attempt, setAttempt] = useState(0)
  const requestKey = search ? `${JSON.stringify(search)}#${attempt}` : null
  const [outcome, setOutcome] = useState<Outcome | null>(null)

  useEffect(() => {
    if (!search || !requestKey) {
      return
    }
    const controller = new AbortController()
    retrieveChunks(search, controller.signal)
      .then((response) => {
        if (!controller.signal.aborted) {
          setOutcome({ key: requestKey, kind: 'loaded', response })
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setOutcome({ key: requestKey, kind: 'failed', error: asApiError(error) })
        }
      })
    return () => controller.abort()
  }, [search, requestKey])

  const current = outcome && outcome.key === requestKey ? outcome : null
  const loading = requestKey !== null && current === null

  const problem = questionProblem(question)
  const length = questionLength(question)
  const reason = !selectedModel ? 'Choose a model to search with.' : problem
  const questionRef = useRef<HTMLTextAreaElement>(null)
  const ids = { question: useId(), counter: useId(), reason: useId(), topK: useId() }

  function submit(event: FormEvent) {
    event.preventDefault()
    if (reason || !selectedModel) {
      return
    }
    const next = toAddress({ question: question.trim(), model: selectedModel.id, topK })
    if (next.toString() === addressKey) {
      setAttempt((n) => n + 1) // the same search again
    } else {
      setParams(next)
    }
  }

  function fillExample(text: string) {
    setQuestion(text)
    questionRef.current?.focus()
  }

  const status = loading
    ? `Retrieving chunks with ${search?.model}…`
    : current?.kind === 'loaded'
      ? `Found ${current.response.results.length} chunks with ${current.response.model}.`
      : ''

  return (
    <>
      <p className="page-purpose">{PAGES.retrieval.purpose}</p>

      <form className="retrieval-form" onSubmit={submit} noValidate>
        <div className="field field-wide">
          <label htmlFor={ids.question}>Question</label>
          <textarea
            id={ids.question}
            ref={questionRef}
            rows={3}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            aria-describedby={ids.counter}
            aria-invalid={length > QUESTION_MAX_CHARACTERS}
          />
          <span
            id={ids.counter}
            className={length > QUESTION_MAX_CHARACTERS ? 'counter counter-over' : 'counter'}
          >
            {length.toLocaleString('en-GB')} / {QUESTION_MAX_CHARACTERS.toLocaleString('en-GB')}{' '}
            characters
          </span>
        </div>

        <details className="examples">
          <summary>Example questions from the research evaluation</summary>
          <ul className="example-list">
            {EVALUATION_QUESTIONS.map((example) => (
              <li key={example.id}>
                <button
                  type="button"
                  className="example-button"
                  onClick={() => fillExample(example.question)}
                >
                  {example.question}
                </button>
              </li>
            ))}
          </ul>
        </details>

        <fieldset className="model-picker">
          <legend>Embedding model</legend>
          {!selectedModel && <p className="section-note">Choose a model to search with.</p>}
          <div className="card-grid">
            {EMBEDDING_MODELS.map((model) => (
              <ModelOption
                key={model.id}
                model={model}
                checked={selectedModel?.id === model.id}
                onSelect={selectModel}
              />
            ))}
          </div>
          <p className="section-note">
            The choice is shared with the Model Lab and saved in this browser.
          </p>
        </fieldset>

        <div className="form-actions">
          <div className="field">
            <label htmlFor={ids.topK}>Number of results</label>
            <select
              id={ids.topK}
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            >
              {TOP_K_CHOICES.map((choice) => (
                <option key={choice} value={choice}>
                  {choice}
                </option>
              ))}
            </select>
          </div>
          <button
            type="submit"
            className="button-primary"
            disabled={reason !== null}
            aria-describedby={reason ? ids.reason : undefined}
          >
            Retrieve
          </button>
          {reason && (
            <p id={ids.reason} className="form-reason">
              {reason}
            </p>
          )}
        </div>
      </form>

      <p className="visually-hidden" role="status">
        {status}
      </p>

      {loading && (
        <div className="config-panel retrieval-status">
          <p className="status-line">Retrieving…</p>
          <p>The first request for a model can take several seconds while the model loads.</p>
        </div>
      )}

      {current?.kind === 'failed' && (
        <div className="config-panel retrieval-status" role="alert">
          <p className="status-line">
            {current.error.kind === 'unreachable'
              ? 'Backend not available'
              : current.error.kind === 'invalid'
                ? 'The request was not accepted'
                : 'Retrieval failed'}
          </p>
          <p>{current.error.message}</p>
          {current.error.kind === 'unreachable' && (
            <>
              <p>Start it in a second terminal, then try again:</p>
              <pre className="command">{START_COMMAND}</pre>
            </>
          )}
          {current.error.kind !== 'invalid' && (
            <button
              type="button"
              className="button-secondary"
              onClick={() => setAttempt((n) => n + 1)}
            >
              Try again
            </button>
          )}
        </div>
      )}

      {current?.kind === 'loaded' && <RetrievalResults response={current.response} />}
    </>
  )
}

export default RetrievalPage
