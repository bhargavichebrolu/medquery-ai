import type { EmbeddingModel, ModelId } from '../data/embeddingModels'

interface ModelOptionProps {
  model: EmbeddingModel
  checked: boolean
  onSelect: (id: ModelId) => void
}

// A selectable card. It is a real radio button inside a <label>, so clicking
// anywhere on the card selects it and arrow keys move between models.
// Screen readers announce the model name, then the detail as its description.
function ModelOption({ model, checked, onSelect }: ModelOptionProps) {
  const detailId = `model-${model.id}-detail`

  return (
    <label className="model-option">
      <input
        type="radio"
        name="embedding-model"
        value={model.id}
        checked={checked}
        onChange={() => onSelect(model.id)}
        aria-label={model.id}
        aria-describedby={detailId}
      />
      <span className="model-option-name">{model.id}</span>
      <span id={detailId} className="model-option-detail">
        {model.dimensions} dimensions
      </span>
    </label>
  )
}

export default ModelOption
