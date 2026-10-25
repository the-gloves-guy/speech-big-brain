"""Bounded local audio decoding, Silero VAD, and reproducible speech cuts."""

import hashlib
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .extension_data import file_hash, read_rows, write_json

DIRECTORIES = {
    "Samoan": "sam",
    "Nuiean": "niu",
    "Niuean": "niu",
    "Tongan": "ton",
    "Hindi": "hi",
    "Thai": "th",
}
PACIFIC_TARGETS = ["sam", "niu", "ton"]
TARGETS = [*PACIFIC_TARGETS, "hi", "th"]


def speech_chunks(intervals, min_samples=48000, max_samples=160000):
    """Split contiguous speech evenly, keeping time order without overlap."""
    if not 0 < min_samples <= max_samples:
        raise ValueError("Require 0 < minimum <= maximum duration")
    kept, dropped, previous_end = [], [], 0
    for interval in intervals:
        start, end = int(interval["start"]), int(interval["end"])
        if start < previous_end or end <= start:
            raise ValueError("VAD intervals overlap or have invalid boundaries")
        previous_end = end
        size = end - start
        if size < min_samples:
            dropped.append(
                {"start": start, "end": end, "reason": "short_speech_region"}
            )
            continue
        # Balancing avoids throwing away a short tail of an otherwise long turn.
        count = math.ceil(size / max_samples)
        for index in range(count):
            left = start + size * index // count
            right = start + size * (index + 1) // count
            if right - left < min_samples:
                dropped.append({"start": left, "end": right, "reason": "short_split"})
            else:
                kept.append({"start": left, "end": right})
    return kept, dropped


def inventory(root, languages=None):
    required = set(PACIFIC_TARGETS if languages is None else languages)
    if not required or not required.issubset(TARGETS):
        raise ValueError("Unsupported validation languages")
    rows, hashes = [], set()
    for directory, code in DIRECTORIES.items():
        if languages is not None and code not in required:
            continue
        for path in sorted((Path(root) / directory).glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".mp3", ".wav"}:
                continue
            digest = file_hash(path)
            if digest in hashes:
                raise ValueError(f"Duplicate source audio: {path}")
            hashes.add(digest)
            stat = path.stat()
            rows.append(
                {
                    "source": str(path.resolve()),
                    "source_id": f"{code}-{digest[:16]}",
                    "language": code,
                    "sha256": digest,
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    if not required.issubset({row["language"] for row in rows}):
        raise ValueError(
            f"Need WAV/MP3 sources for requested languages: {sorted(required)}"
        )
    return rows


def prepare(root, output, min_seconds=3.0, max_seconds=10.0, languages=None):
    import silero_vad
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    min_seconds, max_seconds = float(min_seconds), float(max_seconds)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = inventory(root, languages)
    weights = Path(silero_vad.__file__).parent / "data/silero_vad.jit"
    configuration = {
        "target_languages": sorted({r["language"] for r in sources}),
        "sample_rate": 16000,
        "minimum_seconds": min_seconds,
        "maximum_seconds": max_seconds,
        "vad": "Silero TorchScript CPU",
        "vad_version": importlib.metadata.version("silero-vad"),
        "vad_weights_sha256": file_hash(weights),
        "threshold": 0.5,
        "vad_min_speech_ms": 250,
        "min_silence_ms": 300,
        "speech_pad_ms": 150,
        "split": "equal contiguous non-overlapping chunks; no concatenation of distant speech",
        "sources": sources,
    }
    identity = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    marker = output / "segments_ready.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] != identity:
            raise ValueError(
                "Input clips or VAD settings changed; choose a new output directory"
            )
        rows = read_rows(output / "segments.jsonl")
        if any(
            not Path(row["path"]).is_file() or file_hash(row["path"]) != row["sha256"]
            for row in rows
        ):
            raise ValueError("A saved speech segment is missing or changed")
        return rows, saved
    context = output / "configuration.json"
    if context.exists() and json.loads(context.read_text()) != configuration:
        raise ValueError("Cannot resume with different inputs or VAD settings")
    write_json(context, configuration)
    decoded = output / "decoded"
    segments = output / "segments"
    decoded.mkdir(exist_ok=True)
    segments.mkdir(exist_ok=True)
    model = load_silero_vad(onnx=False).eval()
    all_rows, summaries = [], []
    for number, source in enumerate(sources, 1):
        summary_path = output / f"{source['source_id']}.json"
        if summary_path.exists():
            saved = json.loads(summary_path.read_text())
            if saved["identity"] != identity or any(
                not Path(row["path"]).is_file() or file_hash(row["path"]) != row["sha256"]
                for row in saved["segments"]
            ):
                raise ValueError("Invalid partial VAD output")
            all_rows.extend(saved["segments"])
            summaries.append(saved["summary"])
            continue
        full = decoded / f"{source['source_id']}.wav"
        if not full.exists():
            temporary = full.with_suffix(".tmp.wav")
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-threads",
                    "1",
                    "-y",
                    "-i",
                    source["source"],
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_f32le",
                    str(temporary),
                ],
                check=True,
            )
            temporary.replace(full)
        signal, rate = sf.read(full, dtype="float32")
        if rate != 16000 or signal.ndim != 1 or not np.isfinite(signal).all():
            raise ValueError(f"Invalid decoded audio: {source['source']}")
        print(f"VAD source {number}/{len(sources)}: {source['source']}", flush=True)
        model.reset_states()
        with torch.inference_mode():
            intervals = get_speech_timestamps(
                torch.from_numpy(signal),
                model,
                sampling_rate=16000,
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=300,
                speech_pad_ms=150,
                return_seconds=False,
            )
        kept, dropped = speech_chunks(
            intervals, round(min_seconds * 16000), round(max_seconds * 16000)
        )
        rows = []
        for interval in kept:
            start, end = interval["start"], interval["end"]
            identifier = f"{source['source_id']}-{start:010d}-{end:010d}"
            path = segments / f"{identifier}.wav"
            sf.write(path, signal[start:end], 16000, subtype="FLOAT")
            rows.append(
                {
                    "id": identifier,
                    "path": str(path),
                    "language": source["language"],
                    "source_id": source["source_id"],
                    "source": source["source"],
                    "group": source["source_id"],
                    "start_seconds": start / 16000,
                    "end_seconds": end / 16000,
                    "duration": (end - start) / 16000,
                    "sha256": file_hash(path),
                }
            )
        summary = {
            **source,
            "decoded_seconds": len(signal) / 16000,
            "vad_regions": len(intervals),
            "vad_region_seconds_with_padding": sum(
                row["end"] - row["start"] for row in intervals
            )
            / 16000,
            "retained_segments": len(rows),
            "retained_seconds": sum(row["duration"] for row in rows),
            "dropped_short_regions": len(dropped),
            "dropped_short_seconds": sum(row["end"] - row["start"] for row in dropped)
            / 16000,
        }
        write_json(
            summary_path,
            {
                "identity": identity,
                "summary": summary,
                "vad_intervals_samples": intervals,
                "dropped_intervals_samples": dropped,
                "segments": rows,
            },
        )
        all_rows.extend(rows)
        summaries.append(summary)
        del signal
    if not all_rows:
        raise ValueError("No speech segments survived the minimum duration")
    with (output / "segments.jsonl").open("w") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in all_rows)
    saved = {
        "identity": identity,
        "configuration": configuration,
        "source_summaries": summaries,
        "segments": len(all_rows),
    }
    write_json(marker, saved)
    return all_rows, saved
