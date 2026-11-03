"""Turn retrieval results into an LLM request (PRD 3.2).

Each retrieved chunk becomes one source document, in rank order and verbatim, titled
"<article title> (chunk N of M)". A source map records which chunk each document position
came from, so citations (which point to a document position) can be traced back to the
retrieved chunk in PRD 3.4. Providers without native citations (the mock, a future local
model) can use `render_plain_text`, which labels the sources [S1]...[Sn].
"""

from dataclasses import dataclass
from functools import cache

from medquery_api.knowledge_base import KnowledgeBase
from medquery_api.llm.base import GenerationRequest, SourceDocument
from medquery_api.llm.mock import NOT_ENOUGH_EVIDENCE
from medquery_api.retrieval import Retrieval

# The research's generation rules (notebook 06), adapted for answers built only from the
# supplied documents. The not-enough-evidence sentence is shared with the mock provider.
SYSTEM_INSTRUCTIONS = f"""\
You are a healthcare information assistant. Answer the user's question using only the NHS \
documents provided with it.

Rules:
1. Base every statement on the provided documents, and cite the passages you rely on.
2. Do not add facts that the documents do not support.
3. Do not diagnose, and do not prescribe or recommend treatment beyond what the documents say.
4. Do not present yourself as a doctor or healthcare professional.
5. If the documents include emergency or urgent-care guidance relevant to the question, \
repeat it accurately and do not change it.
6. If the documents do not contain enough information to answer the question, reply with \
exactly this sentence and nothing else: "{NOT_ENOUGH_EVIDENCE}"
7. Keep the answer clear and concise."""


@dataclass(frozen=True)
class SourceReference:
    """Where one source document came from."""

    index: int  # position in GenerationRequest.documents; Citation.document_index uses this
    label: str  # "S1", "S2", ... as used in the plain-text rendering
    source_file: str
    chunk_id: int
    rank: int
    score: float
    article_title: str
    chunk_count: int  # chunks in the whole article


@dataclass(frozen=True)
class AnswerContext:
    request: GenerationRequest
    sources: tuple[SourceReference, ...]

    def source_for(self, document_index: int) -> SourceReference:
        """The chunk behind a document position (for example, a citation's document_index)."""
        if not 0 <= document_index < len(self.sources):
            raise IndexError(f"no source document at position {document_index}")
        return self.sources[document_index]


@dataclass(frozen=True)
class ArticleInfo:
    title: str
    chunk_count: int


@cache
def article_catalogue(knowledge_base: KnowledgeBase) -> dict[str, ArticleInfo]:
    """Title and chunk count for every article. The title is the article's first line without
    its "# " heading marker (the first line of chunk 1, which is the raw file's first line)."""
    counts: dict[str, int] = {}
    titles: dict[str, str] = {}
    for chunk in knowledge_base.chunks:
        counts[chunk.source_file] = counts.get(chunk.source_file, 0) + 1
        if chunk.chunk_id == 1:
            first_line = chunk.text.splitlines()[0] if chunk.text else ""
            titles[chunk.source_file] = first_line.lstrip("#").strip()
    return {
        source_file: ArticleInfo(
            title=titles.get(source_file) or source_file.removesuffix(".txt"),
            chunk_count=count,
        )
        for source_file, count in counts.items()
    }


def document_title(article_title: str, chunk_id: int, chunk_count: int) -> str:
    return f"{article_title} (chunk {chunk_id} of {chunk_count})"


def build_context(
    retrieval: Retrieval,
    knowledge_base: KnowledgeBase,
    *,
    max_tokens: int = 4096,
) -> AnswerContext:
    """The LLM request for a retrieval: instructions, the retrieved chunks, then the question.

    With no retrieved chunks the request has no documents, and the instructions tell the
    model to say the evidence is not enough.
    """
    catalogue = article_catalogue(knowledge_base)
    hits = sorted(retrieval.hits, key=lambda hit: hit.rank)
    documents: list[SourceDocument] = []
    sources: list[SourceReference] = []
    for index, hit in enumerate(hits):
        chunk = hit.chunk
        article = catalogue.get(
            chunk.source_file,
            ArticleInfo(title=chunk.source_file.removesuffix(".txt"), chunk_count=1),
        )
        title = document_title(article.title, chunk.chunk_id, article.chunk_count)
        documents.append(SourceDocument(title=title, text=chunk.text))
        sources.append(
            SourceReference(
                index=index,
                label=f"S{index + 1}",
                source_file=chunk.source_file,
                chunk_id=chunk.chunk_id,
                rank=hit.rank,
                score=hit.score,
                article_title=article.title,
                chunk_count=article.chunk_count,
            )
        )
    request = GenerationRequest(
        system=SYSTEM_INSTRUCTIONS,
        documents=tuple(documents),
        question=retrieval.question,
        max_tokens=max_tokens,
    )
    return AnswerContext(request=request, sources=tuple(sources))


def render_plain_text(context: AnswerContext) -> str:
    """The sources and question as one text, for providers without native citations.

    Each source is "[S1] <title>" followed by its verbatim text; the model should cite
    sources by these labels.
    """
    blocks = [
        f"[{source.label}] {document.title}\n{document.text}"
        for source, document in zip(context.sources, context.request.documents, strict=True)
    ]
    sources_text = "\n\n".join(blocks) if blocks else "(No documents were retrieved.)"
    return (
        f"SOURCES\n\n{sources_text}\n\n"
        f"QUESTION\n\n{context.request.question}\n\n"
        "Cite the sources you use by their labels, for example [S1]."
    )
