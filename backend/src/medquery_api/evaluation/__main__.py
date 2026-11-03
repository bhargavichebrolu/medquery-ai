"""Check an evaluation dataset file (the bundled research dataset unless a path is given).

    uv run python -m medquery_api.evaluation [PATH]

Prints the dataset's identifier and contents, and checks its relevant articles against the
knowledge base when the research data folder is available. Exits with 1 if anything is wrong.
Only reads files.
"""

import argparse
import sys
from pathlib import Path

from medquery_api import config
from medquery_api.dataset import DataFolderNotFoundError
from medquery_api.evaluation.dataset import (
    DEFAULT_DATASET,
    EvaluationDatasetError,
    article_catalogue,
    load_dataset,
    missing_articles,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m medquery_api.evaluation", description="Check an evaluation dataset file."
    )
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(args.path)
    except EvaluationDatasetError as error:
        print(error, file=sys.stderr)
        return 1

    in_scope, out_of_scope = dataset.in_scope_questions, dataset.out_of_scope_questions
    references = sum(q.reference_answer is not None for q in in_scope)
    print(f"{dataset.name} ({args.path.name})")
    print(f"  identifier      {dataset.identifier}")
    print(f"  format version  {dataset.format_version}; relevance by {dataset.relevance_level}")
    print(
        f"  questions       {len(in_scope)} in scope ({references} with an unverified reference "
        f"answer), {len(out_of_scope)} out of scope"
    )

    try:
        catalogue = article_catalogue(config.data_dir())
    except DataFolderNotFoundError:
        print(
            "  articles        not checked: research data folder not found "
            "(restore Healthcare_RAG/data or set MEDQUERY_DATA_DIR)"
        )
        return 0
    problems = missing_articles(dataset, catalogue)
    if problems:
        print(EvaluationDatasetError(problems, str(args.path)), file=sys.stderr)
        return 1
    print(
        f"  articles        all {len(dataset.relevant_articles)} relevant articles are among the "
        f"{len(catalogue)} in the knowledge base"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
