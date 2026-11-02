"""Command line: download models, or validate embeddings against the research archive.

uv run python -m medquery_api.embeddings download [MODEL ...]
uv run python -m medquery_api.embeddings validate [--model MODEL ...] [--save-dir PATH]
"""

import argparse
import json
import sys
from pathlib import Path

from medquery_api import config
from medquery_api.embeddings.models import MODELS, get_model


def download(names: list[str]) -> int:
    from medquery_api.embeddings.encoder import load_sentence_transformer

    for name in names:
        spec = get_model(name)
        print(f"Downloading {name} ({spec.model_id} at {spec.revision[:12]}) if not cached...")
        load_sentence_transformer(spec, "cpu", local_files_only=False)
        print(f"  {name}: ready in the local model cache")
    return 0


def validate(names: list[str], save_dir: Path | None, as_json: bool) -> int:
    from medquery_api.embeddings.validation import SIMILARITY_THRESHOLD, validate_model

    data_dir = config.data_dir()
    reports = []
    for name in names:
        report = validate_model(name, data_dir, save_dir=save_dir)
        reports.append(report)
        if not as_json:
            print(
                f"\n{name}: {'PASS' if report['passed'] else 'FAIL'}\n"
                f"  chunks {report['chunks']} (match archive: {report['chunks_match_archive']}), "
                f"size {report['dimension']}, all finite: {report['all_finite']}\n"
                f"  largest |length - 1| {report['max_length_deviation']:.3e}\n"
                f"  cosine with archive: min {report['cosine_min']:.9f}, "
                f"mean {report['cosine_mean']:.9f} (threshold {SIMILARITY_THRESHOLD})\n"
                f"  largest 1 - cosine {report['max_one_minus_cosine']:.3e}, "
                f"largest single-number difference {report['max_abs_difference']:.3e}\n"
                f"  nearest archived vector correct: {report['nearest_correct']} of "
                f"{report['chunks']}\n"
                f"  chunks over the {report['token_limit']}-token limit: "
                f"{report['chunks_over_token_limit']} (longest {report['longest_chunk_tokens']})\n"
                f"  embedding time {report['embedding_seconds']} s, "
                f"peak memory {report['peak_memory_mib']:.0f} MiB"
            )
            for failure in report["failures"]:
                print(f"  FAILURE: {failure}")
    if as_json:
        print(json.dumps(reports, indent=2))
    return 0 if all(report["passed"] for report in reports) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m medquery_api.embeddings")
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="download models into the local cache")
    get.add_argument("models", nargs="*", default=list(MODELS), help="default: all four")
    check = commands.add_parser("validate", help="compare with the archived embeddings")
    check.add_argument("--model", action="append", dest="models", help="default: all four")
    check.add_argument("--save-dir", type=Path, help="also save vectors (not in Healthcare_RAG)")
    check.add_argument("--json", action="store_true", help="print the reports as JSON")
    args = parser.parse_args(argv)

    if args.command == "download":
        return download(args.models)
    return validate(args.models or list(MODELS), args.save_dir, args.json)


if __name__ == "__main__":
    sys.exit(main())
