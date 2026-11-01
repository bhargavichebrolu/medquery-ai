// The four embedding models from the original research (Healthcare_RAG/).
// Every value was verified read-only against the archive on 2026-11-01:
// - Hugging Face IDs: model names loaded in notebook 04.
// - Dimensions and vector counts: the FAISS index files in data/faiss/<Model>/.
// - Normalised: normalize_embeddings=True in every encode call (notebooks 04-06),
//   and every stored vector has length 1.
// - Prefixes: E5 only; "passage: " when embedding chunks (notebook 04) and
//   "query: " when embedding questions (notebooks 05-06).
// These describe how the models were configured, not how well they performed.

export type ModelId = 'MPNet' | 'MiniLM' | 'BGE' | 'E5'

export interface EmbeddingModel {
  id: ModelId
  huggingFaceId: string
  dimensions: number
  queryPrefix: string | null
  passagePrefix: string | null
}

export const EMBEDDING_MODELS: EmbeddingModel[] = [
  {
    id: 'MPNet',
    huggingFaceId: 'sentence-transformers/all-mpnet-base-v2',
    dimensions: 768,
    queryPrefix: null,
    passagePrefix: null,
  },
  {
    id: 'MiniLM',
    huggingFaceId: 'sentence-transformers/all-MiniLM-L6-v2',
    dimensions: 384,
    queryPrefix: null,
    passagePrefix: null,
  },
  {
    id: 'BGE',
    huggingFaceId: 'BAAI/bge-small-en-v1.5',
    dimensions: 384,
    queryPrefix: null,
    passagePrefix: null,
  },
  {
    id: 'E5',
    huggingFaceId: 'intfloat/multilingual-e5-base',
    dimensions: 768,
    queryPrefix: 'query: ',
    passagePrefix: 'passage: ',
  },
]

// Settings shared by all four models in the research.
export const RESEARCH_SETUP = {
  normalised: 'Yes (every stored vector has length 1)',
  vectorsIndexed: '3,796',
  indexType: 'FAISS IndexFlatIP (exact search, inner product)',
  topK: 5, // every TOP_K constant in notebook 06
  generator: 'Llama 3.1 8B Instruct', // meta-llama/Llama-3.1-8B-Instruct in notebook 06
}

export function isModelId(value: unknown): value is ModelId {
  return EMBEDDING_MODELS.some((model) => model.id === value)
}

export function findModel(id: ModelId): EmbeddingModel {
  return EMBEDDING_MODELS.find((model) => model.id === id)!
}
