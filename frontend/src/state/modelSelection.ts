import { createContext, useContext } from 'react'
import type { EmbeddingModel, ModelId } from '../data/embeddingModels'

// What every component can read (or change) about the selected model.
export interface ModelSelection {
  selectedModel: EmbeddingModel | null
  selectModel: (id: ModelId) => void
}

// The browser storage key the choice is saved under.
export const STORAGE_KEY = 'medquery.selectedModel'

export const ModelSelectionContext = createContext<ModelSelection | null>(null)

// Any component inside <ModelSelectionProvider> can call this to get the selection.
export function useSelectedModel(): ModelSelection {
  const selection = useContext(ModelSelectionContext)
  if (!selection) {
    throw new Error('useSelectedModel must be used inside <ModelSelectionProvider>')
  }
  return selection
}
