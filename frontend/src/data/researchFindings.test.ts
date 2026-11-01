import { describe, expect, it } from 'vitest'
import {
  ARTICLE_COUNT,
  ASTHMA_QUESTION_COUNT,
  DIFFERING_QUESTIONS,
  IMPERFECT_QUESTIONS,
  QUESTION_COUNT,
  questionsWhereModelsDiffer,
} from './researchFindings'

// These pin the statements shown on the Analytics page to the verified research data.
describe('research findings', () => {
  it('describes the small evaluation set', () => {
    expect(QUESTION_COUNT).toBe(10)
    expect(ARTICLE_COUNT).toBe(8)
    expect(ASTHMA_QUESTION_COUNT).toBe(3)
  })

  it('locates where the models fall short and where they differ', () => {
    expect(IMPERFECT_QUESTIONS).toEqual([6, 7, 9])
    expect(DIFFERING_QUESTIONS).toEqual([7, 9])
  })

  it('confirms BGE and E5 are identical and MPNet leads them by one question', () => {
    expect(questionsWhereModelsDiffer('BGE', 'E5')).toEqual([])
    expect(questionsWhereModelsDiffer('MPNet', 'BGE')).toEqual([7])
  })
})
