import { useState } from 'react'
import { Link, useLocation, useSearchParams } from 'react-router'
import ArticleTags from '../components/ArticleTags'
import {
  CONTENTS_PAGE_COUNT,
  EVALUATION_CONTENTS_PAGES,
  FILTER_LABELS,
  SORT_LABELS,
  TOTAL_CHUNKS,
  parseFilter,
  parseSort,
  selectArticles,
  type ArticleFilter,
  type ArticleSort,
} from '../data/articleCatalogue'
import { ARTICLES } from '../data/knowledgeBase'
import { PAGES } from '../navigation'

const PAGE_SIZE = 50

function listQuestions(ids: number[]) {
  return ids.length === 1
    ? `question ${ids[0]}`
    : `questions ${ids.slice(0, -1).join(', ')} and ${ids.at(-1)}`
}

function KnowledgeBasePage() {
  // Search, filter and sort live in the address (?q=…&show=…&sort=…), so they survive
  // a refresh, can be bookmarked, and come back when returning from an article.
  const [searchParams, setSearchParams] = useSearchParams()
  const query = searchParams.get('q') ?? ''
  const filter = parseFilter(searchParams.get('show'))
  const sort = parseSort(searchParams.get('sort'))
  const location = useLocation()

  // How many matching rows are shown; resets whenever the search or filters change.
  const [shown, setShown] = useState({ forSearch: location.search, count: PAGE_SIZE })
  const visibleCount = shown.forSearch === location.search ? shown.count : PAGE_SIZE

  const matching = selectArticles(query, filter, sort)
  const visible = matching.slice(0, visibleCount)

  function update(name: string, value: string, defaultValue: string) {
    const next = new URLSearchParams(searchParams)
    if (value === defaultValue) {
      next.delete(name)
    } else {
      next.set(name, value)
    }
    setSearchParams(next, { replace: true })
  }

  const flaggedDiffered = EVALUATION_CONTENTS_PAGES.filter((e) => e.modelsDiffered)
  const flaggedSame = EVALUATION_CONTENTS_PAGES.filter((e) => !e.modelsDiffered)

  return (
    <>
      <p className="page-purpose">{PAGES.knowledgeBase.purpose}</p>

      <section className="notice" aria-labelledby="kb-source-heading">
        <h2 id="kb-source-heading">Original research dataset (read-only)</h2>
        <p>
          {ARTICLES.length} NHS articles split into {TOTAL_CHUNKS.toLocaleString('en-GB')} chunks
          of up to 1,000 characters with a 200-character overlap. Titles are each article&apos;s
          first line. The NHS page addresses were not recorded, so no links to nhs.uk are shown.
          This page lists article metadata. To search the chunks by meaning, use{' '}
          <Link to={PAGES.retrieval.path} className="text-link">
            Retrieval
          </Link>
          ; reading every article&apos;s full text and adding documents are not built yet.
        </p>
      </section>

      <section className="info-panel" aria-labelledby="kb-quality-heading">
        <h2 id="kb-quality-heading">Dataset classification: contents pages</h2>
        <p>
          {CONTENTS_PAGE_COUNT} articles are classified as <strong>contents page only</strong>: a
          short description followed by links (“About…”, “Side effects”, “How and when to take…”)
          whose pages were not collected. The rule: an article is classified this way if its text
          contains all three of those links. This describes the data; it is not a measure of
          retrieval quality.
        </p>
        <p className="after-paragraph">
          {EVALUATION_CONTENTS_PAGES.length} of the 8 evaluation articles are classified this way:{' '}
          {EVALUATION_CONTENTS_PAGES.map((e, i) => (
            <span key={e.article.id}>
              {i > 0 && (i === EVALUATION_CONTENTS_PAGES.length - 1 ? ' and ' : ', ')}
              {e.article.title} ({listQuestions(e.questionIds)})
            </span>
          ))}
          . The models disagreed on {listQuestions(flaggedDiffered.flatMap((e) => e.questionIds))}
          {flaggedSame.length > 0 && (
            <>
              , but all ranked the relevant article first on{' '}
              {listQuestions(flaggedSame.flatMap((e) => e.questionIds))}
            </>
          )}
          . The thin articles may have played a part in the weaker results, but this small
          evaluation cannot show that.
        </p>
      </section>

      <section className="dashboard-section" aria-labelledby="kb-articles-heading">
        <h2 id="kb-articles-heading">Articles</h2>

        <div className="toolbar">
          <label className="field">
            <span>Search titles</span>
            <input
              type="search"
              value={query}
              placeholder="e.g. asthma"
              onChange={(event) => update('q', event.target.value, '')}
            />
          </label>

          <fieldset className="segmented">
            <legend>Show</legend>
            {(Object.keys(FILTER_LABELS) as ArticleFilter[]).map((option) => (
              <label key={option}>
                <input
                  type="radio"
                  name="article-filter"
                  value={option}
                  checked={filter === option}
                  onChange={() => update('show', option, 'all')}
                />
                <span>{FILTER_LABELS[option]}</span>
              </label>
            ))}
          </fieldset>

          <label className="field">
            <span>Sort</span>
            <select value={sort} onChange={(event) => update('sort', event.target.value, 'title')}>
              {(Object.keys(SORT_LABELS) as ArticleSort[]).map((option) => (
                <option key={option} value={option}>
                  {SORT_LABELS[option]}
                </option>
              ))}
            </select>
          </label>
        </div>

        <p className="count-line" aria-live="polite">
          Showing {visible.length} of {matching.length} articles
          {matching.length !== ARTICLES.length && ` (${ARTICLES.length} in total)`}
        </p>

        {matching.length === 0 ? (
          <div className="placeholder">
            <p>No articles match your search.</p>
          </div>
        ) : (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Title</th>
                  <th scope="col">File</th>
                  <th scope="col" className="numeric">
                    Chunks
                  </th>
                  <th scope="col" className="numeric">
                    Characters
                  </th>
                </tr>
              </thead>
              <tbody>
                {visible.map((article) => (
                  <tr key={article.id}>
                    <th scope="row">
                      <Link
                        to={`${PAGES.knowledgeBase.path}/${article.id}`}
                        state={{ catalogueSearch: location.search }}
                        className="table-link"
                      >
                        {article.title}
                      </Link>
                      <ArticleTags article={article} />
                    </th>
                    <td>
                      <code>{article.id}.txt</code>
                    </td>
                    <td className="numeric">{article.chunkLengths.length}</td>
                    <td className="numeric">{article.characters.toLocaleString('en-GB')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {visible.length < matching.length && (
          <button
            type="button"
            className="button-secondary"
            onClick={() => setShown({ forSearch: location.search, count: visibleCount + PAGE_SIZE })}
          >
            Show {Math.min(PAGE_SIZE, matching.length - visible.length)} more
          </button>
        )}
      </section>
    </>
  )
}

export default KnowledgeBasePage
