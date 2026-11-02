import ModelOption from '../components/ModelOption'
import { EMBEDDING_MODELS, RESEARCH_SETUP, type EmbeddingModel } from '../data/embeddingModels'
import { PAGES } from '../navigation'
import { useSelectedModel } from '../state/modelSelection'

function describePrefixes(model: EmbeddingModel) {
  if (!model.queryPrefix && !model.passagePrefix) {
    return 'None (text is embedded as it is)'
  }
  return (
    <>
      <code>{model.queryPrefix}</code> before questions, <code>{model.passagePrefix}</code> before
      document chunks
    </>
  )
}

function ModelLabPage() {
  const { selectedModel, selectModel } = useSelectedModel()

  return (
    <>
      <p className="page-purpose">{PAGES.modelLab.purpose}</p>

      <section className="info-panel" aria-labelledby="explainer-heading">
        <h2 id="explainer-heading">What is an embedding model?</h2>
        <p>
          An embedding model turns a piece of text into a list of numbers (a vector), so that texts
          with similar meanings end up close together. MedQuery uses it to find the document chunks
          most relevant to a question. Writing the answer is a separate job for a large language
          model (LLM); the original research used {RESEARCH_SETUP.generator}.
        </p>
      </section>

      <fieldset className="model-picker">
        <legend>Choose an embedding model</legend>
        <div className="card-grid">
          {EMBEDDING_MODELS.map((model) => (
            <ModelOption
              key={model.id}
              model={model}
              checked={selectedModel?.id === model.id}
              onSelect={selectModel}
            />
          ))}
        </div>
        <p className="section-note">
          Your choice is saved in this browser only. The Retrieval page sends it to the local
          backend with each question.
        </p>
      </fieldset>

      <section className="dashboard-section" aria-labelledby="config-heading">
        <h2 id="config-heading">
          Selected configuration{selectedModel ? `: ${selectedModel.id}` : ''}
        </h2>
        {selectedModel ? (
          <div className="config-panel">
            <dl className="config-list">
              <dt>Hugging Face ID</dt>
              <dd>
                <code>{selectedModel.huggingFaceId}</code>
              </dd>
              <dt>Dimensions</dt>
              <dd>{selectedModel.dimensions}</dd>
              <dt>Normalised</dt>
              <dd>{RESEARCH_SETUP.normalised}</dd>
              <dt>Text prefixes</dt>
              <dd>{describePrefixes(selectedModel)}</dd>
              <dt>Vectors indexed</dt>
              <dd>{RESEARCH_SETUP.vectorsIndexed}</dd>
              <dt>Index</dt>
              <dd>{RESEARCH_SETUP.indexType}</dd>
            </dl>
            <p className="stat-source">
              Source: original research (notebooks 04–06 and the FAISS index files)
            </p>
          </div>
        ) : (
          <div className="placeholder">
            <p className="placeholder-label">No model selected</p>
            <p>Choose a model above to see its configuration.</p>
          </div>
        )}
      </section>

      <section className="placeholder" aria-labelledby="retrieval-heading">
        <h3 id="retrieval-heading" className="placeholder-label">
          Retrieval settings (read-only)
        </h3>
        <p>
          The original research retrieved the {RESEARCH_SETUP.topK} most similar chunks for each
          question (top-k = {RESEARCH_SETUP.topK}). On the Retrieval page you can choose from 1
          to 20.
        </p>
      </section>
    </>
  )
}

export default ModelLabPage
