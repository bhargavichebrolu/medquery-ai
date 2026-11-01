// Statements the Analytics page makes about the research results, calculated from
// the exact data so the wording can never drift from the numbers.
import type { ModelId } from './embeddingModels'
import { EVALUATION_QUESTIONS, QUESTION_RESULTS } from './researchResults'

// The reciprocal rank (MRR@5 for a single question) a model scored on one question.
export function resultFor(model: ModelId, questionId: number): number {
  return QUESTION_RESULTS.find((r) => r.model === model && r.questionId === questionId)!.mrrAt5
}

function allResultsFor(questionId: number): number[] {
  return QUESTION_RESULTS.filter((r) => r.questionId === questionId).map((r) => r.mrrAt5)
}

const questionIds = EVALUATION_QUESTIONS.map((q) => q.id)

export const QUESTION_COUNT = EVALUATION_QUESTIONS.length

export const ARTICLE_COUNT = new Set(EVALUATION_QUESTIONS.map((q) => q.relevantArticle)).size

export const ASTHMA_QUESTION_COUNT = EVALUATION_QUESTIONS.filter(
  (q) => q.relevantArticle === 'asthma.txt',
).length

// Questions where at least one model did not rank the relevant article first.
export const IMPERFECT_QUESTIONS = questionIds.filter((id) => allResultsFor(id).some((rr) => rr < 1))

// Questions where the models did not all get the same result.
export const DIFFERING_QUESTIONS = questionIds.filter((id) => new Set(allResultsFor(id)).size > 1)

// Questions where two models' results differ.
export function questionsWhereModelsDiffer(a: ModelId, b: ModelId): number[] {
  return questionIds.filter((id) => resultFor(a, id) !== resultFor(b, id))
}
