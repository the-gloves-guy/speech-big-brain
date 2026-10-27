"""Audited comparisons, matched diagnostics and a portable Large v3 report."""

import base64
import csv
import html
import json
import shutil
from types import SimpleNamespace

from .extension_data import CODES, file_hash, read_rows, write_json
from .speaker_pilot import bind
from .whisper_small_results import portable_figure, read_scores, reuse_baselines
from .youtube_evaluate import write_csv


def baselines(study):
    reuse_baselines(study)
    expected = json.loads((study.root / "data_ready.json").read_text())
    small = json.loads((study.small_reference / "data_ready.json").read_text())
    if expected["cohorts"] != small["cohorts"]:
        raise ValueError(
            "Large/Small cohort identities differ; rerun matched baselines"
        )
    files = {}
    ids = [r["id"] for r in study.rows("external")]
    for root in [study.reference, study.small_reference]:
        directory = root / "frozen"
        cal = json.loads((directory / "calibration.json").read_text())
        if cal["model_sha256"] != file_hash(directory / "model.pt"):
            raise ValueError("Comparator calibration/checkpoint mismatch")
        for cohort in ["fleurs_test", "new_test", "external"]:
            read_scores(
                directory / f"{cohort}.npz", [r["id"] for r in study.rows(cohort)]
            )
        read_scores(directory / "external.npz", ids)
        for name in [
            "model.pt",
            "calibration.json",
            "benchmarks.json",
            "fleurs_test.npz",
            "new_test.npz",
            "external.npz",
        ]:
            path = directory / name
            files[str(path)] = file_hash(path)
    for name in [
        "model.pt",
        "calibration.json",
        "report.json",
        "evaluation_complete.json",
    ]:
        path = study.pacific / "partial" / name
        files[str(path)] = file_hash(path)
    for name in [
        "rows.jsonl",
        "complete.json",
        "duration.csv",
        "gain.csv",
        "predictions.csv",
    ]:
        path = study.small_reference / "diagnostics" / name
        files[str(path)] = file_hash(path)
    bind(
        study.root / "best_comparators.json",
        {
            "files_sha256": files,
            "policy": "read-only; no model or calibration refitting",
        },
    )


def audit_comparators(study):
    for marker in ["reused_comparators.json", "best_comparators.json"]:
        saved = json.loads((study.root / marker).read_text())
        for name, checksum in saved["files_sha256"].items():
            if file_hash(name) != checksum:
                raise ValueError(f"Reused comparator changed: {name}")


def additional_external_models(study):
    audit_comparators(study)
    result = {}
    for name, root in [
        ("whisper_tiny15_frozen", study.reference),
        ("whisper_small15_frozen", study.small_reference),
    ]:
        scores, codes = read_scores(
            root / "frozen/external.npz", [r["id"] for r in study.rows("external")]
        )
        temperature = json.loads((root / "frozen/calibration.json").read_text())[
            "temperature"
        ]
        result[name] = scores, codes, temperature
    return result


def diagnostics(study):
    from .whisper_extension_diagnostics import summarize
    from .whisper_large import LargeAudioAdapter

    audit_comparators(study)
    source = study.small_reference / "diagnostics"
    records = read_rows(source / "rows.jsonl")
    metadata = json.loads((source / "complete.json").read_text())
    if file_hash(source / "rows.jsonl") != metadata["rows_sha256"]:
        raise ValueError("Matched diagnostic rows changed")
    selection = {
        r["id"] for c in ["vox_validation", "new_selection"] for r in study.rows(c)
    }
    sources = {r["source_id"] for r in study.rows("external")}
    for row in records:
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Diagnostic waveform changed")
        if row["probe"] == "gain" and row["parent_id"] not in selection:
            raise ValueError("Gain probe outside selection split")
        if row["probe"] == "duration" and row["source_id"] not in sources:
            raise ValueError("Duration probe outside external snapshot")
    directory = study.root / "diagnostics"
    directory.mkdir(exist_ok=True)
    bind(
        directory / "identity.json",
        {
            "rows_sha256": file_hash(source / "rows.jsonl"),
            "model_sha256": file_hash(study.root / "frozen/model.pt"),
        },
    )
    if not (directory / "rows.jsonl").exists():
        shutil.copyfile(source / "rows.jsonl", directory / "rows.jsonl")
    adapter = LargeAudioAdapter(study.root, study.root / "frozen/model.pt")
    values = study.score_adapter(
        adapter, records, directory / "large_frozen.npz", "large_diagnostics"
    )
    adapter.release()
    temperature = json.loads((study.root / "frozen/calibration.json").read_text())[
        "temperature"
    ]
    summarize(
        directory,
        records,
        {study.extension_prefix + "_frozen": (values, CODES, temperature)},
    )
    for name in ["duration.csv", "gain.csv", "predictions.csv"]:
        with (directory / name).open() as stream:
            current = list(csv.DictReader(stream))
        with (source / name).open() as stream:
            archived = [
                r
                for r in csv.DictReader(stream)
                if r["model"] in study.comparison_models
            ]
        write_csv(directory / name, current + archived)
    write_json(
        directory / "complete.json",
        {
            "rows": len(records),
            "rows_sha256": file_hash(directory / "rows.jsonl"),
            "models": list(study.comparison_models),
            "duration_languages": metadata["duration_languages"],
            "no_matched_niuean_duration_windows": metadata[
                "no_matched_niuean_duration_windows"
            ],
            "gain_only_on_selection": True,
        },
    )


def report(study):
    from .whisper_extension_results import external_results
    from .whisper_large_report import build

    audit_comparators(study)
    proxy = SimpleNamespace(
        root=study.root,
        config={**study.config, "external": str(study.root / "external")},
        rows=study.rows,
        methods=study.methods,
        extension_prefix=study.extension_prefix,
        comparison_models=study.comparison_models,
        additional_external_models=study.additional_external_models,
    )
    external_results(proxy)
    build(study)


def render_pdf(html_path, pdf_path):
    import subprocess

    chrome = shutil.which("google-chrome")
    if not chrome:
        raise RuntimeError("Headless Chrome is required for the PDF report")
    result = subprocess.run(
        [
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
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"PDF rendering failed: {result.stderr[-6000:]}")
    text = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if "Whisper Large v3" not in text or "Complete plot appendix" not in text:
        raise ValueError("Large v3 PDF is missing analysis or figures")
    return text


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def embedded_data(value):
    return base64.b64encode(json.dumps(value).encode()).decode()


def safe(value):
    return html.escape(str(value))


def figure(path):
    return portable_figure(path, str(path))
