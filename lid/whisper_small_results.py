"""Provenance-checked comparator reuse and focused Small15 diagnostics/reporting."""

import base64
import csv
import html
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .extension_data import CODES, file_hash, read_rows, write_json
from .speaker_pilot import bind, digest
from .whisper_extension_diagnostics import summarize
from .whisper_small import SmallAdapter
from .youtube_evaluate import write_csv


def read_scores(path, identifiers):
    with np.load(path) as saved:
        if saved["ids"].tolist() != identifiers:
            raise ValueError(f"Comparator segment IDs differ: {path}")
        scores = saved["logits"]
        codes = saved["codes"].tolist()
        if len(codes) != len(set(codes)) or not set(CODES[:12]).issubset(codes):
            raise ValueError("Comparator language vocabulary is invalid")
        if (
            scores.shape != (len(identifiers), len(codes))
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Invalid comparator score matrix")
    return scores, codes


def reuse_baselines(study):
    expected = json.loads((study.reference / "data_ready.json").read_text())
    actual = json.loads((study.root / "data_ready.json").read_text())
    for name, meta in actual["cohorts"].items():
        if meta["sha256"] != expected["cohorts"][name]["sha256"]:
            raise ValueError(f"Small/Tiny cohort mismatch: {name}")
    files = {}
    external = Path(study.config["external"])
    context = json.loads(
        (external / "speechbrain_checkpoint_identity.json").read_text()
    )
    for name, checksum in context.items():
        if file_hash(name) != checksum:
            raise ValueError("Archived SpeechBrain checkpoint or calibration changed")
        files[name] = checksum
    base = Path(
        json.loads((study.reference / "configuration.json").read_text())["base"]
    )
    ids = [r["id"] for r in study.rows("external")]
    for method in [
        "whisper_native",
        "whisper_filtered",
        "whisper_frozen",
        "whisper_finetuned",
        "speechbrain_finetuned",
    ]:
        directory = study.reference / "baselines" / method
        identity = json.loads((directory / "identity.json").read_text())
        checkpoint = (
            base / method / "model.pt"
            if method not in ["whisper_native", "whisper_filtered"]
            else base / "pretrained/tiny.pt"
        )
        if identity["checkpoint"] != file_hash(checkpoint) or identity[
            "calibration"
        ] != file_hash(base / method / "calibration.json"):
            raise ValueError("Archived twelve-language control changed")
        read_scores(directory / "external.npz", ids)
        for filename in ["external.npz", "calibration.json", "identity.json"]:
            p = directory / filename
            files[str(p)] = file_hash(p)
    for method in ["frozen", "finetuned"]:
        directory = study.reference / method
        cal = json.loads((directory / "calibration.json").read_text())
        if cal["model_sha256"] != file_hash(directory / "model.pt"):
            raise ValueError("Tiny15 calibration/checkpoint mismatch")
        read_scores(directory / "external.npz", ids)
        for filename in ["external.npz", "calibration.json", "benchmarks.json"]:
            p = directory / filename
            files[str(p)] = file_hash(p)
    target = study.root / "baselines"
    if not target.exists():
        target.symlink_to(
            (study.reference / "baselines").resolve(), target_is_directory=True
        )
    if target.resolve() != (study.reference / "baselines").resolve():
        raise ValueError("Small comparator directory points to different controls")
    output = study.root / "external"
    output.mkdir(exist_ok=True)
    for filename in ["logits.npz", "model_provenance.json"]:
        source = external / filename
        path = output / filename
        if not path.exists():
            path.symlink_to(source.resolve())
        if file_hash(path) != file_hash(source):
            raise ValueError("Reused external reference changed")
        files[str(source)] = file_hash(source)
    files[str(external / "segments_ready.json")] = file_hash(
        external / "segments_ready.json"
    )
    bind(
        study.root / "reused_comparators.json",
        {
            "files_sha256": files,
            "external_ids_sha256": digest(ids),
            "policy": "read-only reuse of original Tiny/SpeechBrain predictions; no refitting",
        },
    )


def additional_external_models(study):
    saved = json.loads((study.root / "reused_comparators.json").read_text())
    for name, checksum in saved["files_sha256"].items():
        if file_hash(name) != checksum:
            raise ValueError("A reused comparator changed after audit")
    ids = [r["id"] for r in study.rows("external")]
    result = {}
    for method in ["frozen", "finetuned"]:
        directory = study.reference / method
        scores, codes = read_scores(directory / "external.npz", ids)
        temperature = json.loads((directory / "calibration.json").read_text())[
            "temperature"
        ]
        result[f"whisper_tiny15_{method}"] = (scores, codes, temperature)
    directory = study.base / "whisper_frozen"
    calibration = json.loads((directory / "calibration.json").read_text())
    if calibration["model_sha256"] != file_hash(directory / "model.pt"):
        raise ValueError("Small12 calibration/checkpoint mismatch")
    scores, codes = read_scores(directory / "external.npz", ids)
    result["whisper_small12_warm_start"] = (scores, codes, calibration["temperature"])
    return result


def diagnostics(study):
    source = study.reference / "diagnostics"
    manifest = source / "rows.jsonl"
    previous = json.loads((source / "complete.json").read_text())
    if file_hash(manifest) != previous["rows_sha256"]:
        raise ValueError("Reference diagnostic rows changed")
    records = read_rows(manifest)
    selection_ids = {
        r["id"]
        for cohort in ["vox_validation", "new_selection"]
        for r in study.rows(cohort)
    }
    source_ids = {r["source_id"] for r in study.rows("external")}
    for row in records:
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Reference diagnostic waveform changed")
        if row["probe"] == "gain" and row["parent_id"] not in selection_ids:
            raise ValueError("Volume probe is outside checkpoint-selection splits")
        if row["probe"] == "duration" and row["source_id"] not in source_ids:
            raise ValueError("Duration probe source is outside current snapshot")
    directory = study.root / "diagnostics"
    directory.mkdir(exist_ok=True)
    bind(
        directory / "identity.json",
        {
            "config": study.config,
            "reference_rows_sha256": file_hash(manifest),
            "rows": digest(records),
            "data_sha256": file_hash(study.root / "data_ready.json"),
        },
    )
    target = directory / "rows.jsonl"
    if not target.exists():
        shutil.copyfile(manifest, target)
    if file_hash(target) != file_hash(manifest):
        raise ValueError("Small diagnostic rows differ")
    models = {}
    for method in ["frozen", "finetuned"]:
        adapter = SmallAdapter(study.root, study.root / method / "model.pt")
        scores = study.score_adapter(
            adapter,
            records,
            directory / f"whisper_small15_{method}.npz",
            f"small_diagnostics_{method}",
        )
        adapter.release()
        temperature = json.loads(
            (study.root / method / "calibration.json").read_text()
        )["temperature"]
        models[f"whisper_small15_{method}"] = (scores, CODES, temperature)
    summarize(directory, records, models)
    for filename in ["duration.csv", "gain.csv", "predictions.csv"]:
        with (directory / filename).open(newline="") as stream:
            current = list(csv.DictReader(stream))
        with (source / filename).open(newline="") as stream:
            archived = list(csv.DictReader(stream))
        for row in archived:
            if row["model"].startswith("whisper15_"):
                row["model"] = row["model"].replace("whisper15_", "whisper_tiny15_", 1)
        write_csv(directory / filename, current + archived)
    write_json(
        directory / "complete.json",
        {
            "rows_sha256": file_hash(target),
            "rows": len(records),
            "models": list(models),
            "reused_comparator_rows_sha256": file_hash(manifest),
            "gain_only_on_selection": True,
            "duration_languages": sorted(
                {r["language"] for r in records if r["probe"] == "duration"}
            ),
            "no_matched_niuean_duration_windows": not any(
                r["language"] == "niu" and r["probe"] == "duration" for r in records
            ),
        },
    )


def report(study):
    from .whisper_extension_results import external_results
    from .whisper_small_report import build

    # The comparison writer gets its own destination; it never rewrites v4/Tiny reports.
    proxy = SimpleNamespace(
        root=study.root,
        config={**study.config, "external": str(study.root / "external")},
        rows=study.rows,
        extension_prefix=study.extension_prefix,
        additional_external_models=study.additional_external_models,
    )
    external_results(proxy)
    build(study)


def portable_figure(path, caption):
    content = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{content}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def render_pdf(html_path, pdf_path):
    chrome = shutil.which("google-chrome")
    if chrome is None:
        raise RuntimeError("Headless Chrome is required for the requested PDF report")
    command = [
        chrome,
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--disable-background-networking",
        "--renderer-process-limit=1",
        "--no-pdf-header-footer",
        f"--user-data-dir={html_path.parent.resolve() / 'chrome_profile'}",
        f"--print-to-pdf={pdf_path.resolve()}",
        html_path.resolve().as_uri(),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"PDF rendering failed ({result.returncode}): {result.stderr[-6000:]}"
        )
    if not pdf_path.is_file() or pdf_path.stat().st_size < 1000:
        raise RuntimeError("Small PDF report was not created")
    # Verify the report has searchable analysis text and every plot caption.
    extracted = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "Whisper Small15" not in extracted or "Complete plot appendix" not in extracted:
        raise ValueError("Small PDF is missing analysis or plot appendix")
    return extracted
