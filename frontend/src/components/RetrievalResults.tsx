import { Link } from 'react-router'
import type { RetrievalResponse, RetrievedChunk } from '../api'
import { ARTICLES } from '../data/knowledgeBase'
import { formatMs } from '../format'
import { PAGES } from '../navigation'

// Articles by their file name ("asthma.txt"), for titles and links.
const ARTICLES_BY_FILE = new Map(ARTICLES.map((article) => [`${article.id}.txt`, article]))

// A result is tied when its score exactly equals a neighbour's.
function isTied(results: RetrievedChunk[], index: number) {
  const score = results[index].score
  return results[index - 1]?.score === score || results[index + 1]?.score === score
}

interface RetrievalResultsProps {
  response: RetrievalResponse
}

function RetrievalResults({ response }: RetrievalResultsProps) {
  const { results, timings } = response
  const anyTies = results.some((_, index) => isTied(results, index))

  return (
    <section className="dashboard-section retrieval-results" aria-labelledby="results-heading">
      <h2 id="results-heading">Retrieved chunks</h2>
      <p className="section-note">
        The {results.length} most similar chunks to “{response.question}” using{' '}
        <strong>{response.model}</strong> (<code>{response.modelId}</code>, revision{' '}
        <code>{response.revision.slice(0, 7)}</code>). Live results from the research&apos;s own
        index, not the historical evaluation scores shown on Analytics.
      </p>
      <p className="section-note">
        {timings.modelLoadMs > 0 && `Model loaded in ${formatMs(timings.modelLoadMs)} · `}
        Question embedded in {formatMs(timings.embeddingMs)} · searched in{' '}
        {formatMs(timings.searchMs)}
      </p>
      <p className="section-note">
        Scores are cosine similarities between the question and each chunk (1 = the same
        direction). They can be compared within one model, not between models.
        {anyTies &&
          ' Results marked “Tied score” have exactly equal scores and are listed in index order; ' +
            'the original research may have listed tied results in a different order.'}
      </p>

      <ol className="chunk-list">
        {results.map((result, index) => {
          const article = ARTICLES_BY_FILE.get(result.sourceFile)
          return (
            <li key={`${result.sourceFile}#${result.chunkId}`} className="chunk-card result-card">
              <div className="result-head">
                <span className="result-rank">
                  <span className="visually-hidden">Rank </span>
                  {result.rank}
                </span>
                {article ? (
                  <Link to={`${PAGES.knowledgeBase.path}/${article.id}`} className="result-title">
                    {article.title}
                  </Link>
                ) : (
                  <span className="result-title">{result.sourceFile}</span>
                )}
                <span className="result-score">
                  Score{' '}
                  <data value={result.score} title={`Exact score: ${result.score}`}>
                    {result.score.toFixed(4)}
                  </data>
                  <span className="visually-hidden"> (exact value {result.score})</span>
                </span>
                {isTied(results, index) && <span className="tag tag-tied">Tied score</span>}
              </div>
              <p className="chunk-heading">
                <code>{result.sourceFile}</code> · chunk {result.chunkId}
                {article && ` of ${article.chunkLengths.length}`} ·{' '}
                {result.chunkLength.toLocaleString('en-GB')} characters
              </p>
              {/* Verbatim: line breaks kept, and any web addresses stay plain text. */}
              <div className="chunk-text">{result.text}</div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}

export default RetrievalResults
