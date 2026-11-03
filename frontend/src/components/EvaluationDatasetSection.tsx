import { Fragment, useEffect, useState } from 'react'
import { Link } from 'react-router'
import {
  asApiError,
  fetchEvaluationDataset,
  type ApiError,
  type EvaluationDataset,
  type OutOfScopeQuestion,
} from '../api'
import { findArticle } from '../data/articleCatalogue'
import { PAGES } from '../navigation'
import ErrorPanel from './ErrorPanel'

type Outcome =
  | { attempt: number; kind: 'loaded'; dataset: EvaluationDataset }
  | { attempt: number; kind: 'failed'; error: ApiError }

// The approved criteria quote grounding statuses in backticks; show those parts as code.
function WithCode({ text }: { text: string }) {
  return text
    .split('`')
    .map((piece, index) =>
      index % 2 === 1 ? <code key={index}>{piece}</code> : <Fragment key={index}>{piece}</Fragment>,
    )
}

function Statuses({ statuses }: { statuses: string[] }) {
  return statuses.map((status, index) => (
    <Fragment key={status}>
      {index > 0 && ' or '}
      <code>{status}</code>
    </Fragment>
  ))
}

// How the outcome for an out-of-scope question will be decided. "Manual review" is never an
// automatic failure: a person judges the answer against the acceptance criteria.
function Decision({ question }: { question: OutOfScopeQuestion }) {
  if (question.automaticPassStatuses.length === 0) {
    return <>Manual review of every outcome</>
  }
  const otherwise =
    question.otherwise === 'fail'
      ? 'any other outcome fails'
      : 'any other outcome goes to manual review (it does not fail automatically)'
  return (
    <>
      Passes on <Statuses statuses={question.automaticPassStatuses} />; {otherwise}
    </>
  )
}

function ArticleLink({ fileName }: { fileName: string }) {
  const article = findArticle(fileName.replace(/\.txt$/, ''))
  return article ? (
    <Link to={`${PAGES.knowledgeBase.path}/${article.id}`}>
      <code>{fileName}</code>
    </Link>
  ) : (
    <code>{fileName}</code>
  )
}

function DatasetTables({ dataset }: { dataset: EvaluationDataset }) {
  const { inScopeQuestions: inScope, outOfScopeQuestions: outOfScope } = dataset
  return (
    <>
      <dl className="config-list dataset-facts">
        <dt>Name</dt>
        <dd>{dataset.name}</dd>
        <dt>Identifier</dt>
        <dd>
          <code>{dataset.identifier}</code>
        </dd>
        <dt>Format version</dt>
        <dd>{dataset.formatVersion}</dd>
        <dt>Questions</dt>
        <dd>
          {inScope.length} in scope, {outOfScope.length} out of scope
        </dd>
      </dl>

      <h3 id="evaluation-in-scope-heading">In-scope questions</h3>
      <p className="section-note">
        The research's evaluation questions, each with its relevant article. A retrieved chunk
        counts as relevant when it comes from that article. The reference answers were written for
        the research and nobody has recorded checking them: they are <strong>not ground truth</strong>.
      </p>
      <div className="table-scroll dataset-table-box">
        <table className="data-table dataset-table" role="table" aria-labelledby="evaluation-in-scope-heading">
          <thead role="rowgroup">
            <tr role="row">
              <th scope="col" role="columnheader">ID</th>
              <th scope="col" role="columnheader">Question</th>
              <th scope="col" role="columnheader">Relevant article</th>
              <th scope="col" role="columnheader">Reference answer</th>
            </tr>
          </thead>
          <tbody role="rowgroup">
            {inScope.map((q) => (
              <tr key={q.id} role="row">
                <td className="dataset-id" role="cell" data-label="ID">
                  <code>{q.id}</code>
                </td>
                <th scope="row" role="rowheader" data-label="Question">
                  {q.question}
                </th>
                <td role="cell" data-label="Relevant article">
                  {q.relevantArticles.map((fileName, index) => (
                    <Fragment key={fileName}>
                      {index > 0 && ', '}
                      <ArticleLink fileName={fileName} />
                    </Fragment>
                  ))}
                </td>
                <td role="cell" data-label="Reference answer">
                  {q.referenceAnswer ? (
                    <>
                      <span className="tag tag-unverified">Unverified</span> {q.referenceAnswer.text}
                    </>
                  ) : (
                    'None'
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 id="evaluation-out-of-scope-heading">Out-of-scope questions</h3>
      <p className="section-note">
        Questions that should not get an ordinary answer, with the acceptance criteria approved at
        the PRD 3.6 review, word for word.
      </p>
      <div className="table-scroll dataset-table-box">
        <table className="data-table dataset-table" role="table" aria-labelledby="evaluation-out-of-scope-heading">
          <thead role="rowgroup">
            <tr role="row">
              <th scope="col" role="columnheader">ID</th>
              <th scope="col" role="columnheader">Question</th>
              <th scope="col" role="columnheader">Why it is out of scope</th>
              <th scope="col" role="columnheader">Acceptance criteria</th>
              <th scope="col" role="columnheader">How it is decided</th>
            </tr>
          </thead>
          <tbody role="rowgroup">
            {outOfScope.map((q) => (
              <tr key={q.id} role="row">
                <td className="dataset-id" role="cell" data-label="ID">
                  <code>{q.id}</code>
                </td>
                <th scope="row" role="rowheader" data-label="Question">
                  {q.question}
                </th>
                <td role="cell" data-label="Why it is out of scope">
                  {q.reason}
                </td>
                <td role="cell" data-label="Acceptance criteria">
                  <WithCode text={q.acceptanceCriteria} />
                </td>
                <td role="cell" data-label="How it is decided">
                  <Decision question={q} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

// The evaluation dataset (PRD 4.0), read from the backend. It only shows the questions:
// nothing has been run or scored with them.
function EvaluationDatasetSection() {
  const [attempt, setAttempt] = useState(0)
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    fetchEvaluationDataset(controller.signal)
      .then((dataset) => {
        if (!controller.signal.aborted) setOutcome({ attempt, kind: 'loaded', dataset })
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          const failure = asApiError(error, 'Something went wrong while contacting the backend.')
          setOutcome({ attempt, kind: 'failed', error: failure })
        }
      })
    return () => controller.abort()
  }, [attempt])
  const current = outcome?.attempt === attempt ? outcome : null

  return (
    <section className="dashboard-section" aria-labelledby="evaluation-dataset-heading">
      <h2 id="evaluation-dataset-heading">Current application evaluation dataset</h2>
      <p className="section-note">
        The questions this application's own evaluation uses (Part 4), loaded from the backend.
        It is <strong>not a historical research result</strong>: the sections above show the
        original research, and this dataset belongs to the application.{' '}
        <strong>Nothing has been run or scored with it yet.</strong>
      </p>
      {current === null ? (
        <p className="section-note" role="status">
          Loading the evaluation dataset…
        </p>
      ) : current.kind === 'failed' ? (
        <ErrorPanel
          title="Evaluation dataset not available"
          error={current.error}
          onRetry={() => setAttempt((n) => n + 1)}
        />
      ) : (
        <DatasetTables dataset={current.dataset} />
      )}
    </section>
  )
}

export default EvaluationDatasetSection
