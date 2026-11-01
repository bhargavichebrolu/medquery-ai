// Facts about the original NHS research dataset in Healthcare_RAG/.
// These describe the dataset itself, not how well any model performs.
// Each value was verified read-only against the archive on 2026-11-01.
export interface DatasetFact {
  label: string
  value: string
  detail: string
  source: string
}

const SOURCE = 'original research dataset'

export const DATASET_FACTS: DatasetFact[] = [
  {
    // 698 files each in data/raw, data/cleaned and data/chunks, with matching names.
    label: 'NHS articles',
    value: '698',
    detail: 'Source documents in the knowledge base',
    source: SOURCE,
  },
  {
    // 3,796 records across data/chunks/*.json; every model's FAISS index and
    // metadata.json also hold 3,796 entries.
    label: 'Text chunks',
    value: '3,796',
    detail: 'Passages that are embedded and searched',
    source: SOURCE,
  },
  {
    // Model IDs loaded in notebook 04; one folder per model in data/embeddings and data/faiss.
    label: 'Embedding models',
    value: '4',
    detail: 'MPNet, MiniLM, BGE, E5',
    source: SOURCE,
  },
  {
    // RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) in notebook 03.
    label: 'Chunking',
    value: '1,000 characters',
    detail: '200-character overlap between chunks',
    source: SOURCE,
  },
]
