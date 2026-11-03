import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import {
  EVALUATION_QUESTIONS,
  INDEX_BUILDS,
  LATER_RUN_MEAN_GENERATION_SECONDS,
  QUESTION_RESULTS,
  RAG_LATENCY,
  RETRIEVAL_SUMMARY,
} from './researchResults'

// Reads the original research files (read-only). They are tracked in git, so this
// check runs on any copy of the repository.
const RESULTS_DIR = join(dirname(fileURLToPath(import.meta.url)), '../../../Healthcare_RAG/results')

function researchFile(name: string): string {
  return readFileSync(join(RESULTS_DIR, name), 'utf8')
}

// The CSVs used here contain no quoted commas; the column-count check fails loudly if one appears.
function readCsv(name: string): Record<string, string>[] {
  const [header, ...lines] = researchFile(name).trim().split(/\r?\n/)
  const columns = header.split(',')
  return lines.map((line) => {
    const cells = line.split(',')
    expect(cells, `${name}: ${line}`).toHaveLength(columns.length)
    return Object.fromEntries(columns.map((column, i) => [column, cells[i]]))
  })
}

describe('researchResults.ts matches the original research files exactly', () => {
  it('retrieval summary (retrieval_model_comparison.csv)', () => {
    const expected = readCsv('retrieval_model_comparison.csv').map((row) => ({
      model: row['Embedding Model'],
      precisionAt5: Number(row['Precision@5']),
      recallAt5: Number(row['Recall@5']),
      mrrAt5: Number(row['MRR']),
      ndcgAt5: Number(row['nDCG@5']),
    }))
    expect(RETRIEVAL_SUMMARY).toEqual(expected)
  })

  it('all 40 per-question results (corrected_retrieval_results.csv)', () => {
    const rows = readCsv('corrected_retrieval_results.csv')
    expect(rows).toHaveLength(40)
    const expected = rows.map((row) => ({
      model: row.model,
      questionId: Number(row.question_id),
      precisionAt5: Number(row.precision_at_5),
      recallAt5: Number(row.recall_at_5),
      mrrAt5: Number(row.mrr),
      ndcgAt5: Number(row.ndcg_at_5),
    }))
    expect(QUESTION_RESULTS).toEqual(expected)

    // The question text in the per-question file matches the evaluation question list.
    for (const row of rows) {
      const question = EVALUATION_QUESTIONS.find((q) => q.id === Number(row.question_id))
      expect(question?.question).toBe(row.question)
    }
  })

  it('evaluation questions and relevant articles (formal_rag_evaluation_data.json)', () => {
    const records: { question_id: number; question: string; relevant_sources: string[] }[] = JSON.parse(
      researchFile('formal_rag_evaluation_data.json'),
    )
    const expected = records.map((record) => {
      expect(record.relevant_sources).toHaveLength(1)
      return { id: record.question_id, question: record.question, relevantArticle: record.relevant_sources[0] }
    })
    expect(EVALUATION_QUESTIONS).toEqual(expected)
  })

  it('index build times (faiss_experimental_summary.csv)', () => {
    const expected = readCsv('faiss_experimental_summary.csv').map((row) => ({
      model: row.Model,
      vectors: Number(row.Vectors),
      dimension: Number(row.Dimension),
      indexType: row.Index_Type,
      buildTimeSeconds: Number(row.Build_Time_Seconds),
    }))
    expect(INDEX_BUILDS).toEqual(expected)
  })

  it('latency (rag_performance_summary.csv) and the later run (formal_rag_evaluation_data.json)', () => {
    const expected = readCsv('rag_performance_summary.csv').map((row) => ({
      stage: row.Metric,
      meanSeconds: Number(row['Mean (seconds)']),
      stdDevSeconds: Number(row['Std Dev (seconds)']),
      minSeconds: Number(row['Minimum (seconds)']),
      maxSeconds: Number(row['Maximum (seconds)']),
    }))
    expect(RAG_LATENCY).toEqual(expected)

    const records: { generation_time_seconds: number }[] = JSON.parse(
      researchFile('formal_rag_evaluation_data.json'),
    )
    const mean = records.reduce((sum, r) => sum + r.generation_time_seconds, 0) / records.length
    expect(LATER_RUN_MEAN_GENERATION_SECONDS).toBeCloseTo(mean, 12)
  })
})

// The backend's evaluation dataset (PRD 4.0) holds the same 10 research questions. It is tracked
// in git, so this check runs on any copy of the repository and the two copies cannot drift.
const DATASET_FILE = join(
  dirname(fileURLToPath(import.meta.url)),
  '../../../backend/src/medquery_api/evaluation/datasets/research_evaluation.json',
)

interface DatasetQuestion {
  archive_question_id?: number | null
  question: string
  relevant_articles: string[]
}

describe('EVALUATION_QUESTIONS matches the backend evaluation dataset', () => {
  it('has the same research questions, in the same order, with the same relevant articles', () => {
    const dataset: { in_scope_questions: DatasetQuestion[] } = JSON.parse(readFileSync(DATASET_FILE, 'utf8'))
    // Only the research's own questions; any later in-scope questions have no research number.
    const research = dataset.in_scope_questions.filter((q) => q.archive_question_id != null)

    expect(research).toHaveLength(10)
    expect(
      research.map((q) => ({ id: q.archive_question_id, question: q.question, articles: q.relevant_articles })),
    ).toEqual(EVALUATION_QUESTIONS.map((q) => ({ id: q.id, question: q.question, articles: [q.relevantArticle] })))
  })
})
