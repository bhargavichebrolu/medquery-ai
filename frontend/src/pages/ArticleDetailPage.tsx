import { Link, useLocation, useParams } from 'react-router'
import ArticleTags from '../components/ArticleTags'
import { findArticle, questionsUsing } from '../data/articleCatalogue'
import { EVALUATION_CHUNK_TEXT } from '../data/knowledgeBase'
import { PAGES } from '../navigation'

// Return to the catalogue with the search and filters the reader came from, if any.
function useCatalogueLink() {
  const state = useLocation().state as { catalogueSearch?: unknown } | null
  const search = typeof state?.catalogueSearch === 'string' ? state.catalogueSearch : ''
  return `${PAGES.knowledgeBase.path}${search.startsWith('?') ? search : ''}`
}

function ArticleDetailPage() {
  // :articleId in the route /knowledge-base/:articleId, e.g. "asthma".
  const { articleId } = useParams()
  const article = findArticle(articleId)
  const backLink = useCatalogueLink()

  if (!article) {
    return (
      <>
        <Link to={backLink} className="back-link">
          ← All articles
        </Link>
        <h2 className="article-title">Article not found</h2>
        <p className="page-purpose">
          There is no article with the ID <code>{articleId}</code> in the knowledge base.
        </p>
      </>
    )
  }

  const chunkText = EVALUATION_CHUNK_TEXT[article.id]
  const questions = questionsUsing(article.id)
  const textHasWebAddresses = chunkText?.some((text) => /https?:\/\/|www\./i.test(text)) ?? false

  return (
    <>
      <Link to={backLink} className="back-link">
        ← All articles
      </Link>
      <h2 className="article-title">{article.title}</h2>
      <ArticleTags article={article} />

      <div className="config-panel article-meta">
        <dl className="config-list">
          <dt>File</dt>
          <dd>
            <code>{article.id}.txt</code>
          </dd>
          <dt>Characters</dt>
          <dd>{article.characters.toLocaleString('en-GB')} (cleaned text)</dd>
          <dt>Chunks</dt>
          <dd>{article.chunkLengths.length}</dd>
          <dt>Classification</dt>
          <dd>
            {article.contentsPageOnly
              ? 'Contents page only: a short description and links whose pages were not collected'
              : 'Article'}
          </dd>
          <dt>Evaluation</dt>
          <dd>
            {questions.length === 0 ? (
              'Not used by the evaluation questions'
            ) : (
              <ul className="plain-list">
                {questions.map((q) => (
                  <li key={q.id}>
                    Question {q.id}: {q.question}
                  </li>
                ))}
              </ul>
            )}
          </dd>
        </dl>
        <p className="stat-source">Source: original research dataset (data/raw, data/cleaned, data/chunks)</p>
      </div>

      <section className="dashboard-section" aria-labelledby="chunks-heading">
        <h2 id="chunks-heading">Chunks ({article.chunkLengths.length})</h2>

        {chunkText ? (
          <>
            <p className="section-note">
              Text copied verbatim from <code>data/chunks/{article.id}.json</code> in the research
              archive. Text is included for the 8 evaluation articles only.
              {textHasWebAddresses &&
                ' The original page text includes some web addresses (such as charity sites or ' +
                  'image credits); they are shown as plain text, not links.'}
            </p>
            <ol className="chunk-list">
              {chunkText.map((text, i) => (
                <li key={i} className="chunk-card">
                  <p className="chunk-heading">
                    Chunk {i + 1} · {article.chunkLengths[i].toLocaleString('en-GB')} characters
                  </p>
                  <div className="chunk-text">{text}</div>
                </li>
              ))}
            </ol>
          </>
        ) : (
          <>
            <div className="placeholder">
              <p className="placeholder-label">Text available in Part 2</p>
              <p>
                This article&apos;s chunk text will be available when the knowledge base is
                connected to the backend in Part 2. Only the 8 evaluation articles include text
                for now.
              </p>
            </div>
            <div className="table-scroll chunk-lengths">
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">Chunk</th>
                    <th scope="col" className="numeric">
                      Characters
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {article.chunkLengths.map((length, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td className="numeric">{length.toLocaleString('en-GB')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </>
  )
}

export default ArticleDetailPage
