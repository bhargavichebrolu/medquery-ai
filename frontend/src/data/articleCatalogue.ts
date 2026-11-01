// Searching, filtering and sorting the knowledge base articles, plus how each
// article relates to the evaluation questions. Pure functions over the data files.
import { ARTICLES, type KnowledgeBaseArticle } from './knowledgeBase'
import { DIFFERING_QUESTIONS } from './researchFindings'
import { EVALUATION_QUESTIONS } from './researchResults'

export type ArticleFilter = 'all' | 'evaluation' | 'contents'
export type ArticleSort = 'title' | 'chunks'

export const FILTER_LABELS: Record<ArticleFilter, string> = {
  all: 'All',
  evaluation: 'Used in evaluation',
  contents: 'Contents page only',
}

export const SORT_LABELS: Record<ArticleSort, string> = {
  title: 'Title (A–Z)',
  chunks: 'Most chunks first',
}

// Turn values read from the address bar into a known option, falling back to the default.
export function parseFilter(value: string | null): ArticleFilter {
  return value === 'evaluation' || value === 'contents' ? value : 'all'
}

export function parseSort(value: string | null): ArticleSort {
  return value === 'chunks' ? 'chunks' : 'title'
}

// The evaluation questions whose relevant article is this one.
export function questionsUsing(articleId: string) {
  return EVALUATION_QUESTIONS.filter((q) => q.relevantArticle === `${articleId}.txt`)
}

export function isEvaluationArticle(articleId: string): boolean {
  return questionsUsing(articleId).length > 0
}

export function findArticle(articleId: string | undefined): KnowledgeBaseArticle | undefined {
  return ARTICLES.find((article) => article.id === articleId)
}

export function selectArticles(query: string, filter: ArticleFilter, sort: ArticleSort) {
  const needle = query.trim().toLowerCase()
  const matching = ARTICLES.filter(
    (article) =>
      (filter === 'all' ||
        (filter === 'evaluation' && isEvaluationArticle(article.id)) ||
        (filter === 'contents' && article.contentsPageOnly)) &&
      (needle === '' ||
        article.title.toLowerCase().includes(needle) ||
        article.id.toLowerCase().includes(needle)),
  )
  return matching.sort((a, b) =>
    sort === 'chunks'
      ? b.chunkLengths.length - a.chunkLengths.length || a.title.localeCompare(b.title)
      : a.title.localeCompare(b.title),
  )
}

export const TOTAL_CHUNKS = ARTICLES.reduce((sum, a) => sum + a.chunkLengths.length, 0)

export const CONTENTS_PAGE_COUNT = ARTICLES.filter((a) => a.contentsPageOnly).length

// Evaluation articles classified as contents pages, in question order, noting whether
// the models' results differed on their questions.
export const EVALUATION_CONTENTS_PAGES = ARTICLES.filter(
  (a) => a.contentsPageOnly && isEvaluationArticle(a.id),
)
  .map((article) => {
    const questionIds = questionsUsing(article.id).map((q) => q.id)
    return {
      article,
      questionIds,
      modelsDiffered: questionIds.some((id) => DIFFERING_QUESTIONS.includes(id)),
    }
  })
  .sort((a, b) => a.questionIds[0] - b.questionIds[0])
