// Historical results from the original dissertation research (Healthcare_RAG/results/).
// They were NOT produced by this application. Generated once from the source files and
// checked value-for-value by researchResults.test.ts; values are exact, not rounded.
import type { ModelId } from './embeddingModels'

// MRR is MRR@5: only the top 5 results were retrieved for each question.
export interface RetrievalScores {
  precisionAt5: number
  recallAt5: number
  mrrAt5: number
  ndcgAt5: number
}

// Source: retrieval_model_comparison.csv (the mean of the per-question rows below).
export const RETRIEVAL_SUMMARY: ({ model: ModelId } & RetrievalScores)[] = [
  { model: 'BGE', precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 0.9, ndcgAt5: 0.9261859507142916 },
  { model: 'E5', precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 0.9, ndcgAt5: 0.9261859507142916 },
  { model: 'MPNet', precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 0.95, ndcgAt5: 0.9630929753571458 },
  { model: 'MiniLM', precisionAt5: 0.18, recallAt5: 0.9, mrrAt5: 0.8, ndcgAt5: 0.8261859507142916 },
]

// Source: the evaluation questions and their single relevant article (notebook 06,
// also stored in formal_rag_evaluation_data.json).
export const EVALUATION_QUESTIONS = [
  { id: 1, question: "What are the symptoms of asthma?", relevantArticle: 'asthma.txt' },
  { id: 2, question: "What are the causes of asthma?", relevantArticle: 'asthma.txt' },
  { id: 3, question: "How is asthma diagnosed?", relevantArticle: 'asthma.txt' },
  { id: 4, question: "What is aciclovir used to treat?", relevantArticle: 'aciclovir_zovirax.txt' },
  { id: 5, question: "What are the symptoms of a food allergy?", relevantArticle: 'food_allergy.txt' },
  { id: 6, question: "What are the symptoms of hives?", relevantArticle: 'hives.txt' },
  { id: 7, question: "What is amoxicillin used for?", relevantArticle: 'amoxicillin.txt' },
  { id: 8, question: "What is amlodipine used to treat?", relevantArticle: 'amlodipine.txt' },
  { id: 9, question: "What is allopurinol used to treat?", relevantArticle: 'allopurinol-zyloric_uricto.txt' },
  { id: 10, question: "What is adalimumab used to treat?", relevantArticle: 'adalimumab.txt' },
]

// Source: corrected_retrieval_results.csv (10 questions x 4 models).
export const QUESTION_RESULTS: ({ model: ModelId; questionId: number } & RetrievalScores)[] = [
  { model: 'MiniLM', questionId: 1, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MiniLM', questionId: 2, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MiniLM', questionId: 3, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MiniLM', questionId: 4, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MiniLM', questionId: 5, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MiniLM', questionId: 6, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 0.5, ndcgAt5: 0.6309297535714575 },
  { model: 'MiniLM', questionId: 7, precisionAt5: 0.0, recallAt5: 0.0, mrrAt5: 0.0, ndcgAt5: 0.0 },
  { model: 'MiniLM', questionId: 8, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MiniLM', questionId: 9, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 0.5, ndcgAt5: 0.6309297535714575 },
  { model: 'MiniLM', questionId: 10, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MPNet', questionId: 1, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MPNet', questionId: 2, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MPNet', questionId: 3, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MPNet', questionId: 4, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MPNet', questionId: 5, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MPNet', questionId: 6, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 0.5, ndcgAt5: 0.6309297535714575 },
  { model: 'MPNet', questionId: 7, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MPNet', questionId: 8, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MPNet', questionId: 9, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'MPNet', questionId: 10, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'E5', questionId: 1, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'E5', questionId: 2, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'E5', questionId: 3, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'E5', questionId: 4, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'E5', questionId: 5, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'E5', questionId: 6, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 0.5, ndcgAt5: 0.6309297535714575 },
  { model: 'E5', questionId: 7, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 0.5, ndcgAt5: 0.6309297535714575 },
  { model: 'E5', questionId: 8, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'E5', questionId: 9, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'E5', questionId: 10, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'BGE', questionId: 1, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'BGE', questionId: 2, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'BGE', questionId: 3, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'BGE', questionId: 4, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'BGE', questionId: 5, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'BGE', questionId: 6, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 0.5, ndcgAt5: 0.6309297535714575 },
  { model: 'BGE', questionId: 7, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 0.5, ndcgAt5: 0.6309297535714575 },
  { model: 'BGE', questionId: 8, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'BGE', questionId: 9, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
  { model: 'BGE', questionId: 10, precisionAt5: 0.2, recallAt5: 1.0, mrrAt5: 1.0, ndcgAt5: 1.0 },
]

// Source: faiss_experimental_summary.csv (the Index_File column holds Colab paths and is omitted).
export const INDEX_BUILDS: { model: ModelId; vectors: number; dimension: number; indexType: string; buildTimeSeconds: number }[] = [
  { model: 'MiniLM', vectors: 3796, dimension: 384, indexType: 'IndexFlatIP', buildTimeSeconds: 0.0049 },
  { model: 'MPNet', vectors: 3796, dimension: 768, indexType: 'IndexFlatIP', buildTimeSeconds: 0.0054 },
  { model: 'E5', vectors: 3796, dimension: 768, indexType: 'IndexFlatIP', buildTimeSeconds: 0.0071 },
  { model: 'BGE', vectors: 3796, dimension: 384, indexType: 'IndexFlatIP', buildTimeSeconds: 0.0029 },
]

// Source: rag_performance_summary.csv. One specific run: MPNet + Llama 3.1 8B Instruct, top-k 5,
// the 10 evaluation questions, Google Colab (notebook 06 reports a Tesla T4 GPU).
export const RAG_LATENCY: { stage: string; meanSeconds: number; stdDevSeconds: number; minSeconds: number; maxSeconds: number }[] = [
  { stage: 'Retrieval Time', meanSeconds: 0.01952, stdDevSeconds: 0.0033455775120132683, minSeconds: 0.0154, maxSeconds: 0.0252 },
  { stage: 'Generation Time', meanSeconds: 10.69818, stdDevSeconds: 4.737264346246916, minSeconds: 3.5401, maxSeconds: 18.9132 },
  { stage: 'Total Response Time', meanSeconds: 10.72467, stdDevSeconds: 4.738193891265133, minSeconds: 3.562, maxSeconds: 18.9433 },
]

// Mean generation time of a later run of the same setup, computed from the per-question
// generation_time_seconds in formal_rag_evaluation_data.json. Shows that timings vary between runs.
export const LATER_RUN_MEAN_GENERATION_SECONDS = 8.838960266113281
