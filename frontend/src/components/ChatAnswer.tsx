import { useId, type MouseEvent } from 'react'
import { Link } from 'react-router'
import type { AnswerCitation, AnswerResponse, AnswerSource, CitationCheck } from '../api'
import { ARTICLES } from '../data/knowledgeBase'
import { formatMs } from '../format'
import { PAGES } from '../navigation'

// Articles by their file name ("asthma.txt"), for links to the Knowledge Base.
const ARTICLES_BY_FILE = new Map(ARTICLES.map((article) => [`${article.id}.txt`, article]))

const PROVIDER_NAMES: Record<string, string> = {
  mock: 'Mock provider (no language model)',
  claude: 'Claude (Anthropic)',
}

const percent = (rate: number) => `${Math.round(rate * 100)}%`
const count = (n: number) => n.toLocaleString('en-GB')

// Split a chunk into plain and quoted stretches. Citation positions count characters as
// Python does (code points), so the text is split into code points before slicing.
function highlight(text: string, citations: AnswerCitation[]) {
  const characters = Array.from(text)
  const ranges = citations
    .map((c) => [c.startChar, c.endChar] as const)
    .sort((a, b) => a[0] - b[0])
  const merged: [number, number][] = []
  for (const [start, end] of ranges) {
    const last = merged[merged.length - 1]
    if (last && start <= last[1]) {
      last[1] = Math.max(last[1], end)
    } else {
      merged.push([start, end])
    }
  }
  const pieces: { text: string; quoted: boolean }[] = []
  let position = 0
  for (const [start, end] of merged) {
    if (start > position) pieces.push({ text: characters.slice(position, start).join(''), quoted: false })
    pieces.push({ text: characters.slice(start, end).join(''), quoted: true })
    position = end
  }
  if (position < characters.length) pieces.push({ text: characters.slice(position).join(''), quoted: false })
  return pieces
}

interface AnswerTextProps {
  response: AnswerResponse
  markUnsupported: boolean
  sourceId: (label: string) => string
  onSource: (event: MouseEvent<HTMLAnchorElement>, label: string) => void
}

// The answer as written, part by part, each cited part followed by its source markers. When
// asked, sentences without a valid citation are marked, with a visible label that screen
// readers also read. Sentence positions count code points across all parts.
function AnswerText({ response, markUnsupported, sourceId, onSource }: AnswerTextProps) {
  const { sentences } = response
  const lengths = response.parts.map((part) => Array.from(part.text).length)
  const starts = lengths.map((_, index) => lengths.slice(0, index).reduce((a, b) => a + b, 0))
  return (
    <div className="answer-text">
      {response.parts.map((part, index) => {
        const characters = Array.from(part.text)
        const from = starts[index]
        const to = from + characters.length
        const cuts = new Set([from, to])
        for (const sentence of sentences) {
          if (sentence.start > from && sentence.start < to) cuts.add(sentence.start)
          if (sentence.end > from && sentence.end < to) cuts.add(sentence.end)
        }
        const points = [...cuts].sort((a, b) => a - b)
        const labels = [...new Set(part.citations.map((c) => c.sourceLabel))]
        return (
          <span key={index}>
            {points.slice(0, -1).map((start, i) => {
              const end = points[i + 1]
              const text = characters.slice(start - from, end - from).join('')
              const sentence = sentences.find((s) => s.start <= start && end <= s.end)
              if (!markUnsupported || !sentence || sentence.supported) {
                return text
              }
              return (
                <span key={start}>
                  <span className="unsupported">{text}</span>
                  {end === sentence.end && <span className="tag tag-unsupported">No valid citation</span>}
                </span>
              )
            })}
            {labels.map((label) => (
              <a
                key={label}
                href={`#${sourceId(label)}`}
                className="citation-marker"
                aria-label={`Source ${label}`}
                onClick={(event) => onSource(event, label)}
              >
                {label}
              </a>
            ))}
          </span>
        )
      })}
    </div>
  )
}

// What the grounding status means for this answer, shown before the answer text.
function GroundingNote({ response, headingId }: { response: AnswerResponse; headingId: string }) {
  const { sentences, supportedSentences } = response.citationCheck
  if (response.grounding === 'grounded') {
    return (
      <p className="grounding-note">
        <strong>Supported by the retrieved sources.</strong> Every sentence has at least one valid
        citation. This is a check, not proof that the answer is correct.
      </p>
    )
  }
  if (response.grounding === 'partly_grounded') {
    const unsupported = sentences - supportedSentences
    return (
      <p className="grounding-note check-warning">
        <strong>Partly supported.</strong> {count(unsupported)} of {count(sentences)} sentences have
        no valid citation to the retrieved sources; they are marked “No valid citation”. Treat them
        with caution.
      </p>
    )
  }
  return (
    <div className="check-warning grounding-warning" role="note" aria-labelledby={headingId}>
      <p className="status-line" id={headingId}>
        Not supported by the retrieved sources
      </p>
      <p>
        No sentence in this answer has a valid citation to the retrieved NHS text, so it may come
        from the model’s general knowledge and may be wrong. Do not rely on it: read the sources
        below, or ask a pharmacist or GP.
      </p>
    </div>
  )
}

function CitationChecks({ check }: { check: CitationCheck }) {
  const dropped = check.total - check.valid
  const reasons = [
    [check.dropped.unknownDocument, 'named no retrieved source'],
    [check.dropped.invalidRange, 'pointed outside its source'],
    [check.dropped.textMismatch, 'quoted text that did not match its source'],
  ] as const
  return (
    <div className="config-panel answer-checks">
      <h3 className="subsection-heading">Citation checks</h3>
      <dl className="config-list">
        <dt>Valid citations</dt>
        <dd>
          {check.validityRate === null
            ? 'The answer has no citations.'
            : `${count(check.valid)} of ${count(check.total)} (${percent(check.validityRate)})`}
        </dd>
        <dt>Dropped citations</dt>
        <dd>
          {count(dropped)}
          {dropped > 0 &&
            `: ${reasons
              .filter(([n]) => n > 0)
              .map(([n, reason]) => `${count(n)} ${reason}`)
              .join(', ')}`}
        </dd>
        <dt>Grounding coverage</dt>
        <dd>
          {check.groundingCoverage === null
            ? 'No sentences to check.'
            : `${count(check.supportedSentences)} of ${count(check.sentences)} sentences have a valid citation (${percent(check.groundingCoverage)})`}
        </dd>
      </dl>
      <p className="section-note">
        These are checks, not proof. A valid citation means the quoted passage appears word for
        word in the retrieved chunk; it does not show that the answer is correct, complete or
        right for you. Dropped citations are not shown in the answer.
      </p>
    </div>
  )
}

function AnswerDetails({ response }: { response: AnswerResponse }) {
  const { llm, usage, cost, timings } = response
  const cache = usage.cacheReadInputTokens + usage.cacheCreationInputTokens
  return (
    <div className="config-panel answer-details">
      <h3 className="subsection-heading">How this answer was made</h3>
      <dl className="config-list">
        <dt>Language model</dt>
        <dd>
          {PROVIDER_NAMES[llm.provider] ?? llm.provider} · <code>{llm.model}</code>
          {llm.fellBack && ' · a fallback model answered after the requested model declined'}
        </dd>
        <dt>Retrieval</dt>
        <dd>
          {response.embeddingModel}, top {response.topK}
        </dd>
        <dt>Tokens</dt>
        <dd>
          {count(usage.inputTokens)} input · {count(usage.outputTokens)} output
          {cache > 0 &&
            ` · ${count(usage.cacheReadInputTokens)} cache read · ${count(usage.cacheCreationInputTokens)} cache write`}
        </dd>
        <dt>Estimated cost</dt>
        <dd>
          {cost.estimatedUsd === null ? 'Not estimated' : `$${cost.estimatedUsd.toFixed(4)}`}.{' '}
          {cost.note}
        </dd>
        <dt>Time</dt>
        <dd>
          {timings.modelLoadMs > 0 && `model load ${formatMs(timings.modelLoadMs)} · `}
          embedding {formatMs(timings.embeddingMs)} · search {formatMs(timings.searchMs)} ·
          generation {formatMs(timings.generationMs)} · total {formatMs(timings.totalMs)}
        </dd>
      </dl>
    </div>
  )
}

interface SourceCardProps {
  source: AnswerSource
  id: string
  citations: AnswerCitation[]
}

function SourceCard({ source, id, citations }: SourceCardProps) {
  const article = ARTICLES_BY_FILE.get(source.sourceFile)
  return (
    <li id={id} tabIndex={-1} className="chunk-card result-card source-card">
      <div className="result-head">
        <span className="source-label">{source.label}</span>
        {article ? (
          <Link to={`${PAGES.knowledgeBase.path}/${article.id}`} className="result-title">
            {source.articleTitle}
          </Link>
        ) : (
          <span className="result-title">{source.articleTitle}</span>
        )}
        <span className="result-score">
          Rank {source.rank} · Score{' '}
          <data value={source.score} title={`Exact score: ${source.score}`}>
            {source.score.toFixed(4)}
          </data>
          <span className="visually-hidden"> (exact value {source.score})</span>
        </span>
        {citations.length > 0 && <span className="tag">Cited</span>}
      </div>
      <p className="chunk-heading">
        <code>{source.sourceFile}</code> · chunk {source.chunkId} of {source.chunkCount}
      </p>
      {/* Verbatim: line breaks kept, and any web addresses stay plain text. */}
      <div className="chunk-text">
        {highlight(source.text, citations).map((piece, index) =>
          piece.quoted ? <mark key={index}>{piece.text}</mark> : piece.text,
        )}
      </div>
    </li>
  )
}

// One answer: its text with citation markers, the citation checks, how it was made, and the
// sources it was given. The markers link to the source cards below.
function ChatAnswer({ response }: { response: AnswerResponse }) {
  const prefix = useId()
  const sourceId = (label: string) => `${prefix}-source-${label}`
  const citationsFor = (label: string) =>
    response.parts.flatMap((part) => part.citations.filter((c) => c.sourceLabel === label))

  function goToSource(event: MouseEvent<HTMLAnchorElement>, label: string) {
    event.preventDefault() // keep the address unchanged; move focus to the card instead
    const card = document.getElementById(sourceId(label))
    card?.focus()
    card?.scrollIntoView?.({ block: 'nearest' })
  }

  const { grounding } = response
  const hasText = response.parts.some((part) => part.text.trim() !== '')
  // Refusals and "not enough evidence" are messages, not answers, so there is nothing to check.
  const isAnswer = grounding !== 'refused' && grounding !== 'not_enough_evidence'

  return (
    <section className="dashboard-section chat-answer" aria-labelledby={`${prefix}-heading`}>
      <h2 id={`${prefix}-heading`}>Answer</h2>
      <p className="section-note">
        “{response.question}”, answered from the{' '}
        {response.sources.length === 1 ? 'chunk' : `${response.sources.length} chunks`} retrieved
        with {response.embeddingModel}.
      </p>

      {grounding === 'refused' ? (
        <div className="config-panel answer-status">
          <p className="status-line">No answer</p>
          <p>
            The language model declined to answer this question
            {response.refusalCategory && ` (category: ${response.refusalCategory})`}, so no
            answer is shown. Try asking in a different way.
          </p>
        </div>
      ) : grounding === 'not_enough_evidence' ? (
        <div className="config-panel answer-status">
          <p className="status-line">Not enough evidence</p>
          <p>
            The retrieved NHS text does not contain enough information to answer this question, so
            no answer is given. Try rephrasing it, or retrieve more chunks or with another embedding
            model.
          </p>
        </div>
      ) : hasText ? (
        <>
          <GroundingNote response={response} headingId={`${prefix}-grounding`} />
          <AnswerText
            response={response}
            markUnsupported={grounding !== 'grounded'}
            sourceId={sourceId}
            onSource={goToSource}
          />
        </>
      ) : (
        <div className="config-panel answer-status">
          <p className="status-line">No answer</p>
          <p>The language model returned no text.</p>
        </div>
      )}

      {response.stopReason === 'max_tokens' && (
        <p className="section-note answer-cut-off">
          The answer reached the length limit and may be cut off.
        </p>
      )}

      {isAnswer && <CitationChecks check={response.citationCheck} />}
      <AnswerDetails response={response} />

      <h3 className="subsection-heading">Sources the model was given</h3>
      <p className="section-note">
        The retrieved chunks, verbatim, in rank order. Highlighted text is quoted by a valid
        citation. Scores are cosine similarities between the question and each chunk; they can be
        compared within one embedding model, not between models.
      </p>
      <ol className="chunk-list">
        {response.sources.map((source) => (
          <SourceCard
            key={source.label}
            source={source}
            id={sourceId(source.label)}
            citations={citationsFor(source.label)}
          />
        ))}
      </ol>
    </section>
  )
}

export default ChatAnswer
