from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package finetuning outputs into a zip archive and optionally trigger a Colab download."
    )
    parser.add_argument(
        "--outputs_dir",
        default="packages/finetuning/outputs",
        help="Directory containing finetuning outputs.",
    )
    parser.add_argument(
        "--archive_path",
        default="packages/finetuning/finetuning_outputs.zip",
        help="Path for the generated zip archive.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Trigger google.colab.files.download() when running inside Colab.",
    )
    return parser.parse_args()


def collect_artifacts(outputs_dir: Path) -> list[Path]:
    artifacts: list[Path] = []
    for path in outputs_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name == ".gitkeep":
            continue
        artifacts.append(path)
    return sorted(artifacts)


def create_archive(outputs_dir: Path, archive_path: Path, artifacts: list[Path]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in artifacts:
            archive.write(artifact, artifact.relative_to(outputs_dir.parent))


def maybe_download(archive_path: Path) -> None:
    try:
        from google.colab import files  # type: ignore
    except Exception:
        print("Colab download helper is unavailable. Download the archive manually from the file browser.")
        return

    files.download(str(archive_path))


def main() -> int:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).resolve()
    archive_path = Path(args.archive_path).resolve()

    if not outputs_dir.exists():
        print(f"Outputs directory not found: {outputs_dir}")
        return 1

    artifacts = collect_artifacts(outputs_dir)
    if not artifacts:
        print("No finetuning artifacts were found. Refusing to create an empty archive.")
        print("Check that training, evaluation, or comparison actually produced files under outputs/.")
        return 1

    create_archive(outputs_dir, archive_path, artifacts)
    print(f"Created archive: {archive_path}")
    print("Included files:")
    for artifact in artifacts:
        print(f"  - {artifact.relative_to(outputs_dir.parent)}")

    if args.download:
        maybe_download(archive_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
