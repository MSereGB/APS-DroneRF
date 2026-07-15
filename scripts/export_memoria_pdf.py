"""Compila la memoria LaTeX y copia el PDF al directorio de salida."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, TextStringObject

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "informe" / "memoria_final.tex"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "memoria_final_aps_dronerf_entrega.pdf"
AUTHOR = "María Serena Gil"
TITLE = "Detección de actividad de drones a partir de señales RF públicas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def find_tectonic() -> Path | None:
    system_tectonic = shutil.which("tectonic")
    if system_tectonic:
        return Path(system_tectonic)

    configured = os.environ.get("TECTONIC_BIN")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate

    return None


def compile_with_tectonic(source: Path, work_dir: Path, tectonic: Path) -> Path:
    source_arg = source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
    command = [
        str(tectonic),
        "-X",
        "compile",
    ]
    latex_directory = source.parent / "latex"
    if latex_directory.exists():
        hidden = (
            latex_directory.relative_to(ROOT)
            if latex_directory.is_relative_to(ROOT)
            else latex_directory
        )
        command.extend(["--hide", str(hidden)])
    command.extend(
        [
        "--outdir",
        str(work_dir),
        "--outfmt",
        "pdf",
        "--print",
        "--untrusted",
        str(source_arg),
        ]
    )
    subprocess.run(command, cwd=ROOT, check=True)
    return work_dir / f"{source.stem}.pdf"


def compile_with_pdflatex(source: Path, work_dir: Path, pdflatex: str) -> Path:
    source_arg = source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
    command = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={work_dir}",
        str(source_arg),
    ]
    for _ in range(2):
        subprocess.run(command, cwd=ROOT, check=True)
    return work_dir / f"{source.stem}.pdf"


def sha256(path: Path) -> str:
    """Calcular el hash binario de un archivo de entrega."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(output: Path) -> Path:
    """Devolver la ruta del comprobante que acompaña al PDF exportado."""

    return output.with_suffix(".provenance.json")


def normalize_pdf_metadata(output: Path) -> None:
    """Conservar únicamente título y autora en el PDF final."""

    reader = PdfReader(output)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    writer._info = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Title"): TextStringObject(TITLE),
                NameObject("/Author"): TextStringObject(AUTHOR),
            }
        )
    )
    if "/Metadata" in writer._root_object:
        del writer._root_object["/Metadata"]

    temporary = output.with_suffix(".metadata.tmp.pdf")
    with temporary.open("wb") as handle:
        writer.write(handle)
    temporary.replace(output)


def write_provenance(source: Path, output: Path) -> Path:
    """Guardar hashes que vinculan la memoria fuente con su PDF final."""

    source_path = source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
    output_path = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_path": source_path.as_posix(),
        "source_sha256": sha256(source),
        "output_path": output_path.as_posix(),
        "output_sha256": sha256(output),
        "pages": len(PdfReader(str(output)).pages),
    }
    destination = provenance_path(output)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(f"No se encontró la memoria: {source}")

    work_dir = ROOT / "tmp" / "pdfs" / "memoria_final"
    work_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    tectonic = find_tectonic()
    pdflatex = shutil.which("pdflatex")
    if tectonic is not None:
        compiled = compile_with_tectonic(source, work_dir, tectonic)
    elif pdflatex:
        compiled = compile_with_pdflatex(source, work_dir, pdflatex)
    else:
        raise RuntimeError(
            "No se encontró Tectonic ni pdflatex. Instalá un compilador LaTeX "
            "o definí la variable de entorno TECTONIC_BIN."
        )

    shutil.copy2(compiled, output)
    normalize_pdf_metadata(output)
    print(f"PDF de memoria guardado en {output}")
    provenance = write_provenance(source, output)
    print(f"Proveniencia de memoria guardada en {provenance}")


if __name__ == "__main__":
    main()
