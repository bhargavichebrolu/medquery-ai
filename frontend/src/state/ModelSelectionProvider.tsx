import { useCallback, useMemo, useState, type ReactNode } from 'react'
import { findModel, isModelId, type ModelId } from '../data/embeddingModels'
import { ModelSelectionContext, STORAGE_KEY } from './modelSelection'

// Read the saved choice. Anything missing, unknown or unreadable means "no model selected".
function readSavedModelId(): ModelId | null {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    return isModelId(saved) ? saved : null
  } catch {
    return null
  }
}

interface ModelSelectionProviderProps {
  children: ReactNode
}

// Holds the selected model and shares it with every component inside it.
function ModelSelectionProvider({ children }: ModelSelectionProviderProps) {
  const [selectedId, setSelectedId] = useState<ModelId | null>(readSavedModelId)

  const selectModel = useCallback((id: ModelId) => {
    setSelectedId(id)
    try {
      localStorage.setItem(STORAGE_KEY, id)
    } catch {
      // Storage can be blocked (for example in some private windows). The choice
      // still works until the page is refreshed.
    }
  }, [])

  const selection = useMemo(
    () => ({ selectedModel: selectedId ? findModel(selectedId) : null, selectModel }),
    [selectedId, selectModel],
  )

  return <ModelSelectionContext value={selection}>{children}</ModelSelectionContext>
}

export default ModelSelectionProvider
