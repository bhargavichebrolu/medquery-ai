import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { ARTICLES, CONTENTS_PAGE_MARKERS, EVALUATION_CHUNK_TEXT } from './knowledgeBase'
import { EVALUATION_QUESTIONS } from './researchResults'

// Healthcare_RAG/data is deliberately not in git, so the archive comparison can only
// run where the folder exists (the researcher's machine). Elsewhere it is skipped.
// MEDQUERY_DATA_DIR can point the test at the data folder somewhere else.
const DATA_DIR =
  process.env.MEDQUERY_DATA_DIR ??
  join(dirname(fileURLToPath(import.meta.url)), '../../../Healthcare_RAG/data')
const ARCHIVE_PRESENT = existsSync(join(DATA_DIR, 'chunks'))

it('reports whether the archive comparison runs', () => {
  if (!ARCHIVE_PRESENT) {
    console.warn(
      `SKIPPED: research data folder not found at ${DATA_DIR}, so knowledgeBase.ts was not ` +
        'compared with the archive. Run the tests where Healthcare_RAG/data exists to validate it.',
    )
  }
  expect(typeof ARCHIVE_PRESENT).toBe('boolean')
})

// Read exactly as stored (line endings kept); count Unicode code points, as the generator did.
const read = (...parts: string[]) => readFileSync(join(DATA_DIR, ...parts), 'utf8')
const codePoints = (text: string) => [...text].length

interface ArchivedChunk {
  chunk_id: number
  source_file: string
  chunk_length: number
  text: string
}

describe('knowledgeBase.ts is internally consistent', () => {
  it('has 698 articles, 3,796 chunks and 107 contents-page-only articles', () => {
    expect(ARTICLES).toHaveLength(698)
    expect(ARTICLES.reduce((sum, a) => sum + a.chunkLengths.length, 0)).toBe(3796)
    expect(ARTICLES.filter((a) => a.contentsPageOnly)).toHaveLength(107)
    expect(new Set(ARTICLES.map((a) => a.id)).size).toBe(698)
  })

  it('includes chunk text for exactly the 8 evaluation articles, matching their chunk lengths', () => {
    const evaluationIds = [...new Set(EVALUATION_QUESTIONS.map((q) => q.relevantArticle.replace(/\.txt$/, '')))]
    expect(Object.keys(EVALUATION_CHUNK_TEXT).sort()).toEqual(evaluationIds.sort())

    for (const [id, texts] of Object.entries(EVALUATION_CHUNK_TEXT)) {
      const article = ARTICLES.find((a) => a.id === id)!
      expect(texts.map(codePoints)).toEqual(article.chunkLengths)
    }
  })

  it('has no URLs in the article metadata', () => {
    expect(JSON.stringify(ARTICLES)).not.toMatch(/https?:|www\.|nhs\.uk/i)
  })

  it('has no NHS page addresses in the verbatim chunk text', () => {
    // The archived text does contain other web addresses (charity sites, image credits);
    // they are kept verbatim and shown as plain text. None of them is an nhs.uk address.
    const addresses = Object.values(EVALUATION_CHUNK_TEXT)
      .flat()
      .flatMap((text) => text.match(/https?:\/\/\S+|www\.\S+/gi) ?? [])
    expect(addresses).toHaveLength(14)
    expect(addresses.filter((address) => /nhs\.uk/i.test(address))).toEqual([])
  })
})

describe.skipIf(!ARCHIVE_PRESENT)('knowledgeBase.ts matches Healthcare_RAG/data exactly', () => {
  it('lists every article in data/chunks, and only those', () => {
    const ids = readdirSync(join(DATA_DIR, 'chunks'))
      .map((file) => file.replace(/\.json$/, ''))
      .sort()
    expect(ARTICLES.map((a) => a.id)).toEqual(ids)
  })

  it('has the right title, length, chunk lengths and classification for all 698 articles', () => {
    for (const article of ARTICLES) {
      const raw = read('raw', `${article.id}.txt`)
      const firstLine = raw.split(/\r?\n/).find((line) => line.trim())!.trim()
      expect(firstLine, article.id).toBe(`# ${article.title}`)

      expect(article.characters, article.id).toBe(codePoints(read('cleaned', `${article.id}.txt`)))

      const chunks: ArchivedChunk[] = JSON.parse(read('chunks', `${article.id}.json`))
      expect(article.chunkLengths, article.id).toEqual(chunks.map((c) => c.chunk_length))

      const fullText = chunks.map((c) => c.text).join('\n')
      const isContentsPage = CONTENTS_PAGE_MARKERS.every((marker) => fullText.includes(marker))
      expect(article.contentsPageOnly, article.id).toBe(isContentsPage)
    }
  })

  it('copies the evaluation articles’ chunk text verbatim', () => {
    for (const [id, texts] of Object.entries(EVALUATION_CHUNK_TEXT)) {
      const chunks: ArchivedChunk[] = JSON.parse(read('chunks', `${id}.json`))
      expect(texts, id).toEqual(chunks.map((c) => c.text))
    }
  })
})
