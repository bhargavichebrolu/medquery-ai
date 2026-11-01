import { isEvaluationArticle } from '../data/articleCatalogue'
import type { KnowledgeBaseArticle } from '../data/knowledgeBase'

// Text badges (not colour alone) for an article's evaluation use and classification.
function ArticleTags({ article }: { article: KnowledgeBaseArticle }) {
  const evaluation = isEvaluationArticle(article.id)
  if (!evaluation && !article.contentsPageOnly) {
    return null
  }
  return (
    <span className="tag-list">
      {evaluation && <span className="tag tag-evaluation">Evaluation</span>}
      {article.contentsPageOnly && <span className="tag tag-contents">Contents page only</span>}
    </span>
  )
}

export default ArticleTags
