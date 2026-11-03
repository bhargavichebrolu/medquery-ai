"""The evaluation dataset format (PRD 4.0): questions, relevance labels and expected behaviour.

A dataset is one UTF-8 JSON file. `load_dataset` reads and validates it and gives it an
identifier computed from its content, so the same content always has the same identifier and
any change gives a new one. The fields (described in backend/README.md, "Evaluation dataset"):

- `format_version`: 1, the only version this code reads.
- `name`, `description`: text.
- `relevance_level`: "article", the only level (decided for Part 4). A retrieved chunk is
  relevant when its article is one of the question's relevant articles, as in notebook 06.
- `in_scope_questions`: questions the knowledge base can answer, each with its relevant
  articles (file names such as "asthma.txt") and, optionally, a reference answer. Reference
  answers must be marked "unverified": historical text, not ground truth.
- `out_of_scope_questions`: questions that should not get an ordinary answer, each with its
  acceptance criteria, the grounding statuses that pass without review, and what happens
  to any other outcome: "fail", or "manual_review" when a person must judge the answer.

Every problem in a file is reported at once, each saying where it is and what to change.
Relevant articles are checked against the knowledge base separately (`missing_articles`),
because that needs the research data folder.
"""

import hashlib
import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from medquery_api.dataset import DataFolderNotFoundError
from medquery_api.llm.citations import Grounding

FORMAT_VERSION = 1
DEFAULT_DATASET = Path(__file__).parent / "datasets" / "research_evaluation.json"

Otherwise = Literal["fail", "manual_review"]
GROUNDINGS: tuple[str, ...] = get_args(Grounding)
OTHERWISE: tuple[str, ...] = get_args(Otherwise)
RELEVANCE_LEVELS = ("article",)
REFERENCE_STATUSES = ("unverified",)

TOP_FIELDS = (
    "format_version",
    "name",
    "description",
    "relevance_level",
    "in_scope_questions",
    "out_of_scope_questions",
)
IN_SCOPE_FIELDS = ("id", "question", "relevant_articles", "source")
IN_SCOPE_OPTIONAL = ("reference_answer", "archive_question_id")
REFERENCE_FIELDS = ("text", "status", "source")
OUT_OF_SCOPE_FIELDS = (
    "id",
    "question",
    "reason",
    "acceptance_criteria",
    "automatic_pass_statuses",
    "otherwise",
    "source",
)

_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_ARTICLE = re.compile(r"[^/\\]+\.txt")


class EvaluationDatasetError(ValueError):
    """A dataset file that cannot be used, with every problem found in it."""

    def __init__(self, problems: list[str], source: str = "The evaluation dataset") -> None:
        self.problems = problems
        count = f"{len(problems)} problem{'' if len(problems) == 1 else 's'}"
        listed = "\n".join(f"- {problem}" for problem in problems)
        super().__init__(f"{source} is not a valid evaluation dataset ({count}):\n{listed}")


@dataclass(frozen=True)
class ReferenceAnswer:
    text: str
    status: str  # always "unverified" for now
    source: str


@dataclass(frozen=True)
class InScopeQuestion:
    id: str
    question: str
    relevant_articles: tuple[str, ...]
    reference_answer: ReferenceAnswer | None
    archive_question_id: int | None  # the research's question number, when it has one
    source: str


@dataclass(frozen=True)
class OutOfScopeQuestion:
    id: str
    question: str
    reason: str
    acceptance_criteria: str  # the approved wording, word for word
    automatic_pass_statuses: tuple[Grounding, ...]
    otherwise: Otherwise
    source: str


@dataclass(frozen=True)
class EvaluationDataset:
    identifier: str  # "sha256:" and the hash of the content (see dataset_identifier)
    format_version: int
    name: str
    description: str
    relevance_level: str
    in_scope_questions: tuple[InScopeQuestion, ...]
    out_of_scope_questions: tuple[OutOfScopeQuestion, ...]

    @property
    def relevant_articles(self) -> frozenset[str]:
        return frozenset(a for q in self.in_scope_questions for a in q.relevant_articles)


def dataset_identifier(document: object) -> str:
    """A hash of the content, ignoring key order and formatting, so reformatting the file
    keeps its identifier and changing any value gives a new one."""
    canonical = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_dataset(path: Path = DEFAULT_DATASET) -> EvaluationDataset:
    """Read and validate a dataset file. Raises EvaluationDatasetError listing every problem."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise EvaluationDatasetError(["the file does not exist"], str(path)) from None
    except UnicodeDecodeError:
        problem = "the file is not UTF-8 text; save it as UTF-8"
        raise EvaluationDatasetError([problem], str(path)) from None
    try:
        document = json.loads(text, object_pairs_hook=_refuse_repeated_fields)
    except json.JSONDecodeError as error:
        problem = f"not valid JSON: {error.msg} at line {error.lineno}, column {error.colno}"
        raise EvaluationDatasetError([problem], str(path)) from None
    except _RepeatedFieldError as error:
        problem = f'the field "{error.field}" appears twice in one object; keep only one'
        raise EvaluationDatasetError([problem], str(path)) from None
    return parse_dataset(document, str(path))


def parse_dataset(document: object, source: str = "The evaluation dataset") -> EvaluationDataset:
    """Validate an already-parsed JSON document and build the dataset."""
    reader = _Reader()
    top = reader.object(document, "the dataset", TOP_FIELDS)
    if top is None:
        raise EvaluationDatasetError(reader.problems, source)

    version = top.get("format_version")
    if "format_version" in top and (not _is_int(version) or version != FORMAT_VERSION):
        # A different version may lay the file out differently, so stop here.
        reader.problem(
            "format_version",
            f"must be {FORMAT_VERSION}, the only version this backend reads "
            f"(found {json.dumps(version)})",
        )
        raise EvaluationDatasetError(reader.problems, source)

    name = reader.text(top, "name")
    description = reader.text(top, "description")
    level = top.get("relevance_level")
    if "relevance_level" in top and level not in RELEVANCE_LEVELS:
        reader.problem(
            "relevance_level",
            f'must be "article": relevance is labelled by article only, and chunk-level '
            f"labels are not supported (found {json.dumps(level)})",
        )

    in_items = reader.items(top, "in_scope_questions")
    out_items = reader.items(top, "out_of_scope_questions")
    in_scope = tuple(
        question
        for index, item in enumerate(in_items)
        if (question := reader.in_scope(item, f"in_scope_questions[{index}]")) is not None
    )
    out_of_scope = tuple(
        question
        for index, item in enumerate(out_items)
        if (question := reader.out_of_scope(item, f"out_of_scope_questions[{index}]")) is not None
    )
    lists = isinstance(top.get("in_scope_questions"), list) and isinstance(
        top.get("out_of_scope_questions"), list
    )
    if lists and not in_items and not out_items:
        reader.problem("the dataset", "has no questions; add at least one")
    reader.check_unique(top)

    if reader.problems:
        raise EvaluationDatasetError(reader.problems, source)
    return EvaluationDataset(
        identifier=dataset_identifier(document),
        format_version=FORMAT_VERSION,
        name=name,
        description=description,
        relevance_level=level,
        in_scope_questions=in_scope,
        out_of_scope_questions=out_of_scope,
    )


def article_catalogue(data_dir: Path) -> frozenset[str]:
    """The knowledge base's article names: the source_file of every chunk in data/chunks."""
    chunks_dir = data_dir / "chunks"
    if not chunks_dir.is_dir():
        raise DataFolderNotFoundError(f"No chunks folder in {data_dir}")
    names: set[str] = set()
    for path in sorted(chunks_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            names.update(chunk["source_file"] for chunk in json.load(file))
    return frozenset(names)


def missing_articles(dataset: EvaluationDataset, catalogue: Collection[str]) -> list[str]:
    """A problem for each relevant article that is not in the knowledge base."""
    return [
        f'in_scope_questions[{index}] (id "{question.id}"): relevant article "{article}" is not '
        f"one of the {len(catalogue)} articles in the knowledge base; check its spelling "
        '(article names are file names such as "asthma.txt")'
        for index, question in enumerate(dataset.in_scope_questions)
        for article in question.relevant_articles
        if article not in catalogue
    ]


class _RepeatedFieldError(ValueError):
    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.field = field


def _refuse_repeated_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    # json.loads would otherwise keep the last of two fields with the same name, silently.
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _RepeatedFieldError(key)
        result[key] = value
    return result


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _describe(value: object) -> str:
    """The JSON kind of a value, for messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true/false"
    if isinstance(value, int | float):
        return "a number"
    if isinstance(value, str):
        return "text"
    return "a list" if isinstance(value, list) else "an object"


def _normalised(question: str) -> str:
    return " ".join(question.split()).casefold()


class _Reader:
    """Checks one document, collecting every problem instead of stopping at the first."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def problem(self, where: str, message: str) -> None:
        self.problems.append(f"{where}: {message}")

    def object(
        self, value: object, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()
    ) -> dict | None:
        """The value as an object with the required fields and no unknown ones."""
        if not isinstance(value, dict):
            self.problem(where, f"must be an object, not {_describe(value)}")
            return None
        allowed = required + optional
        for field in required:
            if field not in value:
                self.problem(where, f'missing the field "{field}"')
        for field in value:
            if field not in allowed:
                self.problem(where, f'unknown field "{field}" (allowed: {", ".join(allowed)})')
        return value

    def text(self, item: dict, field: str, where: str = "") -> str:
        """A non-empty text field without surrounding spaces (missing fields are reported
        by `object`). `where` is empty for the dataset's own fields."""
        if field not in item:
            return ""
        value = item[field]
        location = f"{where}.{field}" if where else field
        if not isinstance(value, str):
            self.problem(location, f"must be text, not {_describe(value)}")
            return ""
        if not value.strip():
            self.problem(location, "is empty; write the text")
            return ""
        if value != value.strip():
            self.problem(location, "starts or ends with spaces or line breaks; remove them")
        return value

    def items(self, top: dict, field: str) -> list:
        value = top.get(field, [])
        if not isinstance(value, list):
            self.problem(field, f"must be a list, not {_describe(value)}")
            return []
        return value

    def question_object(
        self, value: object, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()
    ) -> tuple[dict | None, str, str]:
        """The question as an object, its id, and `where` naming the id when it has one, so
        every message says which question it is about."""
        identifier = value.get("id") if isinstance(value, dict) else None
        if isinstance(identifier, str) and identifier.strip():
            labelled = f'{where} (id "{identifier}")'
        else:
            identifier, labelled = "", where
        item = self.object(value, labelled, required, optional)
        if item is not None:
            identifier = self.text(item, "id", where)
            if identifier and not _ID.fullmatch(identifier):
                self.problem(
                    f"{where}.id",
                    f'"{identifier}" must be lower-case letters and digits joined by single '
                    'hyphens, such as "research-1"',
                )
        return item, identifier, labelled

    def in_scope(self, value: object, where: str) -> InScopeQuestion | None:
        item, identifier, where = self.question_object(
            value, where, IN_SCOPE_FIELDS, IN_SCOPE_OPTIONAL
        )
        if item is None:
            return None
        question = self.text(item, "question", where)
        articles = self.articles(item, where)
        reference = self.reference(item, where)
        number = item.get("archive_question_id")
        if number is not None and (not _is_int(number) or number < 1):
            self.problem(
                f"{where}.archive_question_id",
                f"must be the research's question number (1 or more), not {json.dumps(number)}",
            )
        source = self.text(item, "source", where)
        return InScopeQuestion(identifier, question, articles, reference, number, source)

    def articles(self, item: dict, where: str) -> tuple[str, ...]:
        if "relevant_articles" not in item:
            return ()
        value = item["relevant_articles"]
        where = f"{where}.relevant_articles"
        if not isinstance(value, list) or not value:
            self.problem(
                where,
                'must be a non-empty list of article file names, such as ["asthma.txt"]',
            )
            return ()
        for index, article in enumerate(value):
            if not isinstance(article, str) or not _ARTICLE.fullmatch(article):
                self.problem(
                    f"{where}[{index}]",
                    f'must be an article file name ending in ".txt", such as "asthma.txt" '
                    f"(found {json.dumps(article)})",
                )
            elif value.index(article) != index:
                self.problem(f"{where}[{index}]", f'"{article}" is listed twice; keep only one')
        return tuple(article for article in value if isinstance(article, str))

    def reference(self, item: dict, where: str) -> ReferenceAnswer | None:
        value = item.get("reference_answer")
        if value is None:
            return None
        where = f"{where}.reference_answer"
        reference = self.object(value, where, REFERENCE_FIELDS)
        if reference is None:
            return None
        status = reference.get("status")
        if "status" in reference and status not in REFERENCE_STATUSES:
            self.problem(
                f"{where}.status",
                f'must be "unverified": no reference answer has been checked, so none can be '
                f"marked otherwise yet (found {json.dumps(status)})",
            )
        return ReferenceAnswer(
            text=self.text(reference, "text", where),
            status=status if isinstance(status, str) else "",
            source=self.text(reference, "source", where),
        )

    def out_of_scope(self, value: object, where: str) -> OutOfScopeQuestion | None:
        item, identifier, where = self.question_object(value, where, OUT_OF_SCOPE_FIELDS)
        if item is None:
            return None
        question = self.text(item, "question", where)
        reason = self.text(item, "reason", where)
        criteria = self.text(item, "acceptance_criteria", where)
        statuses = self.statuses(item, where)
        otherwise = item.get("otherwise")
        if "otherwise" in item and otherwise not in OTHERWISE:
            self.problem(
                f"{where}.otherwise",
                f'must be "fail" or "manual_review" (found {json.dumps(otherwise)})',
            )
        elif "automatic_pass_statuses" in item and not statuses and otherwise == "fail":
            self.problem(
                where,
                "could never pass: it has no automatic_pass_statuses and otherwise is "
                '"fail"; list the statuses that pass, or set otherwise to "manual_review"',
            )
        source = self.text(item, "source", where)
        return OutOfScopeQuestion(
            identifier, question, reason, criteria, statuses, otherwise, source
        )

    def statuses(self, item: dict, where: str) -> tuple[Grounding, ...]:
        if "automatic_pass_statuses" not in item:
            return ()
        value = item["automatic_pass_statuses"]
        where = f"{where}.automatic_pass_statuses"
        if not isinstance(value, list):
            self.problem(where, f"must be a list (it may be empty), not {_describe(value)}")
            return ()
        allowed = ", ".join(GROUNDINGS)
        for index, status in enumerate(value):
            if status not in GROUNDINGS:
                self.problem(
                    f"{where}[{index}]",
                    f"must be a grounding status ({allowed}), not {json.dumps(status)}",
                )
            elif value.index(status) != index:
                self.problem(f"{where}[{index}]", f'"{status}" is listed twice; keep only one')
        return tuple(status for status in value if status in GROUNDINGS)

    def check_unique(self, top: dict) -> None:
        """Every id and every question (ignoring case and spacing) appears once in the dataset."""
        seen_ids: dict[str, str] = {}
        seen_questions: dict[str, str] = {}
        for field in ("in_scope_questions", "out_of_scope_questions"):
            items = top.get(field)
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                where = f"{field}[{index}]"
                identifier, question = item.get("id"), item.get("question")
                if isinstance(identifier, str) and identifier:
                    if identifier in seen_ids:
                        self.problem(
                            where,
                            f'the id "{identifier}" is already used by {seen_ids[identifier]}; '
                            "every question needs its own id",
                        )
                    else:
                        seen_ids[identifier] = where
                if isinstance(question, str) and question.strip():
                    key = _normalised(question)
                    if key in seen_questions:
                        self.problem(
                            where,
                            f"the same question is already at {seen_questions[key]}; "
                            "remove one of them",
                        )
                    else:
                        seen_questions[key] = where
