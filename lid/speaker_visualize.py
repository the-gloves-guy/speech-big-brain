"""Pacific LID geometry, offline point explorer, and speaker-study report."""

import csv
import gc
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .extension_data import BASE, CODES, file_hash, read_rows, write_json
from .extension_data import RUN as ORIGINAL
from .speaker_study import (
    EXTERNAL,
    ROOT,
    SPLITS,
    inventory,
    status,
    write_csv,
)

REPORT = Path("reports/speaker_study")


def load_geometry_embeddings(method, rows):
    import torch

    from .extension_model import cache_embeddings, load_initial, state_fingerprint

    destination = ROOT / f"{method}_lid_embeddings.pt"
    identity = {
        "ids_sha256": hashlib.sha256(
            json.dumps([r["id"] for r in rows]).encode()
        ).hexdigest(),
        "model_sha256": file_hash(ORIGINAL / method / "model.pt"),
        "max_seconds": 10,
        "crop": "center",
    }
    if destination.exists():
        saved = torch.load(destination, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Geometry embedding cache changed")
        return saved["embeddings"].numpy(), identity
    if method == "frozen":
        pieces = []
        for split in SPLITS:
            bundle = torch.load(
                ORIGINAL / f"new_{split}.pt", map_location="cpu", weights_only=True
            )
            expected = read_rows(ORIGINAL / f"new_{split}.jsonl")
            expected_hash = hashlib.sha256(
                json.dumps(expected, sort_keys=True).encode()
            ).hexdigest()
            if bundle["identity"]["rows_sha256"] != expected_hash:
                raise ValueError("Original embedding row order changed")
            pieces.append(bundle["embeddings"])
        external = torch.load(
            EXTERNAL / "base_embeddings.pt", map_location="cpu", weights_only=True
        )
        pieces.append(external["embeddings"])
        embeddings = torch.cat(pieces)
    else:
        encoder, head, _ = load_initial()
        checkpoint = torch.load(
            ORIGINAL / method / "model.pt", map_location="cpu", weights_only=True
        )
        encoder.mods.embedding_model.load_state_dict(checkpoint["encoder"])
        fingerprint = state_fingerprint(encoder.mods.embedding_model)
        del checkpoint, head
        pieces = []
        old_sizes = {
            "selection": len(
                torch.load(
                    BASE / "vox_validation.pt", map_location="cpu", weights_only=True
                )["ids"]
            ),
            "calibration": len(
                torch.load(
                    BASE / "fleurs_direct_validation.pt",
                    map_location="cpu",
                    weights_only=True,
                )["ids"]
            ),
            "test": len(
                torch.load(
                    BASE / "fleurs_direct_test.pt",
                    map_location="cpu",
                    weights_only=True,
                )["ids"]
            ),
        }
        best_epoch = json.loads((ORIGINAL / "partial/complete.json").read_text())[
            "best_epoch"
        ]
        for split in SPLITS:
            selected = read_rows(ORIGINAL / f"new_{split}.jsonl")
            if split == "train":
                piece = cache_embeddings(
                    encoder,
                    selected,
                    ROOT / "partial_radio_train.pt",
                    fingerprint,
                    "lid_partial_train",
                    progress=status,
                )
            else:
                filename = (
                    f"selection_epoch_{best_epoch:02d}.pt"
                    if split == "selection"
                    else f"{split}_embeddings.pt"
                )
                bundle = torch.load(
                    ORIGINAL / "partial" / filename,
                    map_location="cpu",
                    weights_only=True,
                )
                if bundle["identity"]["encoder"] != fingerprint:
                    raise ValueError(
                        "Saved partial cache is from a different checkpoint"
                    )
                piece = bundle["embeddings"][old_sizes[split] :]
                if len(piece) != len(selected):
                    raise ValueError("Saved partial cache shape mismatch")
            pieces.append(piece)
        external = torch.load(
            EXTERNAL / "partial_embeddings.pt", map_location="cpu", weights_only=True
        )
        if external["identity"]["encoder_sha256"] != fingerprint:
            raise ValueError("External encoder fingerprint mismatch")
        pieces.append(external["embeddings"])
        embeddings = torch.cat(pieces)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    if embeddings.shape != (len(rows), 256) or not torch.isfinite(embeddings).all():
        raise ValueError("Invalid LID geometry matrix")
    torch.save({"identity": identity, "embeddings": embeddings}, destination)
    return embeddings.numpy(), identity


def predictions(method, embeddings):
    import torch

    from .extension_model import head_scores
    from .model import new_head

    original = torch.load(BASE / "head/head.pt", map_location="cpu", weights_only=True)
    head = new_head(original["initial_state"], original["indices"])
    head.out = torch.nn.Linear(512, len(CODES))
    checkpoint = torch.load(
        ORIGINAL / method / "model.pt", map_location="cpu", weights_only=True
    )
    head.load_state_dict(checkpoint["head"])
    head.to("cuda").eval()
    logits = head_scores(head, torch.from_numpy(embeddings))
    result = [CODES[i] for i in logits.argmax(-1).tolist()]
    del head, checkpoint, original
    gc.collect()
    torch.cuda.empty_cache()
    return result


def visualize():
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
    from sklearn.preprocessing import normalize
    from umap import UMAP

    REPORT.mkdir(parents=True, exist_ok=True)
    rows = inventory()
    labels = {
        row["id"]: row for row in csv.DictReader((ROOT / "speaker_labels.csv").open())
    }
    train = np.array([r["original_split"] == "train" for r in rows])
    summaries = {}
    for method in ("frozen", "partial"):
        done = ROOT / f"{method}_geometry_complete.json"
        if done.exists():
            saved_config = json.loads(
                (ROOT / f"{method}_geometry_configuration.json").read_text()
            )
            if saved_config["embedding_identity"]["model_sha256"] != file_hash(
                ORIGINAL / method / "model.pt"
            ) or saved_config["inventory_sha256"] != file_hash(
                ROOT / "inventory.jsonl"
            ):
                raise ValueError(
                    "Completed geometry belongs to a different model or inventory"
                )
            summaries[method] = json.loads(done.read_text())
            continue
        embeddings, identity = load_geometry_embeddings(method, rows)
        normalized = normalize(embeddings).astype(np.float32)
        config = {
            "embedding_identity": identity,
            "umap": {
                "n_neighbors": 30,
                "min_dist": 0.1,
                "metric": "cosine",
                "random_state": 42,
                "transform_seed": 42,
                "low_memory": True,
                "n_jobs": 1,
            },
            "fit_split": "original radio train",
            "cluster_space": "L2-normalized embeddings, training-fitted PCA50",
            "hdbscan_min_samples": 10,
            "hdbscan_min_cluster_sizes": [25, 50, 100],
            "inventory_sha256": file_hash(ROOT / "inventory.jsonl"),
        }
        context = ROOT / f"{method}_geometry_configuration.json"
        if context.exists() and json.loads(context.read_text()) != config:
            raise ValueError("Geometry analysis configuration changed")
        write_json(context, config)
        coordinates_path = ROOT / f"{method}_coordinates.npz"
        if coordinates_path.exists():
            with np.load(coordinates_path) as saved:
                coordinates, reduced = saved["coordinates"], saved["pca"]
        else:
            status(
                f"umap_{method}",
                fitting_points=int(train.sum()),
                projecting_points=int((~train).sum()),
            )
            mapper = UMAP(**config["umap"])
            coordinates = np.empty((len(rows), 2), dtype=np.float32)
            coordinates[train] = mapper.fit_transform(normalized[train])
            coordinates[~train] = mapper.transform(normalized[~train])
            pca = PCA(n_components=50, svd_solver="randomized", random_state=42)
            pca.fit(normalized[train])
            reduced = pca.transform(normalized).astype(np.float32)
            np.savez_compressed(
                coordinates_path,
                coordinates=coordinates,
                pca=reduced,
                pca_components=pca.components_,
                pca_mean=pca.mean_,
                ids=np.array([r["id"] for r in rows]),
            )
            del mapper, pca
            gc.collect()
        results, cluster_values = [], {}
        for minimum in (25, 50, 100):
            cluster_path = ROOT / f"{method}_clusters_{minimum}.npy"
            if cluster_path.exists():
                clusters = np.load(cluster_path)
            else:
                status(
                    f"cluster_lid_{method}", min_cluster_size=minimum, points=len(rows)
                )
                clusters = HDBSCAN(
                    min_cluster_size=minimum,
                    min_samples=10,
                    metric="euclidean",
                    n_jobs=2,
                    copy=True,
                ).fit_predict(reduced)
                np.save(cluster_path, clusters)
            cluster_values[minimum] = clusters
            retained = clusters >= 0
            for attribute in ("language", "speaker", "recording", "domain"):
                values = [
                    r["language"]
                    if attribute == "language"
                    else labels.get(r["id"], {}).get("speaker_id", "external_unknown")
                    if attribute == "speaker"
                    else r["group"]
                    if attribute == "recording"
                    else r["domain"]
                    for r in rows
                ]
                mask = retained & (
                    np.array([r["domain"] == "radio" for r in rows])
                    if attribute == "speaker"
                    else True
                )
                result = {
                    "min_cluster_size": minimum,
                    "attribute": attribute,
                    "n": int(mask.sum()),
                    "clusters": len(set(clusters[retained].tolist())),
                    "noise_fraction": float((~retained).mean()),
                    "nmi": float(
                        normalized_mutual_info_score(
                            np.array(values)[mask], clusters[mask]
                        )
                    )
                    if mask.any()
                    else None,
                    "ami": float(
                        adjusted_mutual_info_score(
                            np.array(values)[mask], clusters[mask]
                        )
                    )
                    if mask.any()
                    else None,
                }
                results.append(result)
        write_csv(ROOT / f"{method}_cluster_associations.csv", results)
        predicted = predictions(method, embeddings)
        points = []
        for i, row in enumerate(rows):
            speaker = labels.get(row["id"], {})
            points.append(
                {
                    "id": row["id"],
                    "path": row["path"],
                    "language": row["language"],
                    "domain": row["domain"],
                    "recording": row["group"],
                    "split": row["original_split"],
                    "new_split": speaker.get("new_split", "external"),
                    "speaker": speaker.get("speaker_id", "external_unknown"),
                    "uncertain_speaker": speaker.get("uncertain", "True"),
                    "predicted": predicted[i],
                    "error": "English prediction"
                    if predicted[i] == "en"
                    else "Correct"
                    if predicted[i] == row["language"]
                    else "Other mismatch",
                    "x": float(coordinates[i, 0]),
                    "y": float(coordinates[i, 1]),
                    "cluster": int(cluster_values[50][i]),
                }
            )
        write_csv(ROOT / f"{method}_points.csv", points)
        static_figures(method, coordinates, points)
        explorer(method, points)
        summary = {
            "points": len(rows),
            "radio": int(sum(r["domain"] == "radio" for r in rows)),
            "external": int(sum(r["domain"] == "external" for r in rows)),
            "associations": results,
            "config_sha256": file_hash(context),
        }
        write_json(done, summary)
        summaries[method] = summary
        del embeddings, normalized, reduced
        gc.collect()
    write_json(ROOT / "visualization_complete.json", summaries)


def static_figures(method, coordinates, points):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = REPORT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    attributes = ("language", "domain", "speaker", "cluster", "split", "error")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for axis, attribute in zip(axes.flat, attributes):
        values = np.array([str(p[attribute]) for p in points])
        if attribute == "speaker":
            values = np.array(
                [
                    "Radio: unassigned"
                    if v.startswith("unknown:")
                    else "External: unassigned"
                    if v == "external_unknown"
                    else "Component " + v
                    for v in values
                ]
            )
        elif attribute == "cluster":
            values = np.array(["Noise" if v == "-1" else v for v in values])
        common = [key for key, _ in Counter(values).most_common(8)]
        display = np.array([v if v in common else "Other" for v in values])
        keys = sorted(set(display))
        for index, value in enumerate(keys):
            mask = display == value
            color = plt.get_cmap("tab10")(index % 10)
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=1.5,
                alpha=0.45,
                linewidths=0,
                color=color,
                label=value,
                rasterized=True,
            )
        external = np.array([p["domain"] == "external" for p in points])
        axis.scatter(
            coordinates[external, 0],
            coordinates[external, 1],
            s=12,
            facecolors="none",
            edgecolors="#111111",
            linewidths=0.4,
            rasterized=True,
        )
        axis.set(title=attribute.replace("_", " ").title(), xticks=[], yticks=[])
        axis.legend(markerscale=4, fontsize=6, frameon=False, loc="best")
    fig.suptitle(
        f"Pacific LID embeddings — {'base / frozen' if method == 'frozen' else 'partially fine-tuned'} encoder\n"
        "UMAP fitted on radio training cuts; outlined points are external clips; axes have no physical units",
        fontsize=12,
    )
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            figures / f"{method}_umap.{extension}", dpi=350, bbox_inches="tight"
        )
    plt.close(fig)


def explorer(method, points):
    # Offline canvas: no CDN, telemetry, or third-party browser dependencies.
    rows = [
        {**p, "audio": quote(os.path.relpath(p["path"], REPORT), safe="/")}
        for p in points
    ]
    data = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    template = r"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Pacific LID embedding explorer</title>
<style>body{font:15px system-ui;margin:20px;color:#172033}canvas{width:100%;height:70vh;border:1px solid #ccd;cursor:crosshair}label{margin-right:18px}#info{white-space:pre-wrap}button,select{padding:6px}#legend{font-size:12px;margin:10px 0}</style>
<h1>Pacific LID embeddings: METHOD</h1><p>Scroll to zoom; drag to pan; click a point for its audio reference. Speaker IDs are inferred. Each encoder has its own projection; positions cannot be compared across encoders.</p>
<label>Colour <select id="colour"></select></label><label>Show <select id="filter"><option>all</option><option>radio</option><option>external</option></select></label><button id="reset">Reset view</button>
<div id="legend"></div><canvas id="plot"></canvas><p id="info">Select a point.</p><audio id="audio" controls></audio> <a id="link">Open audio</a>
<script>const points=DATA; const canvas=document.querySelector('#plot'),ctx=canvas.getContext('2d');let sx=1,tx=0,ty=0,drag=null,moved=false;
const attrs=['language','speaker','recording','split','new_split','domain','cluster','error'];const colours=['#0072b2','#d55e00','#009e73','#cc79a7','#e69f00','#56b4e9','#8b4513','#666666','#882255','#44aa99'];
const selector=document.querySelector('#colour'),filter=document.querySelector('#filter');attrs.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k;selector.append(o)});
let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;points.forEach(p=>{xmin=Math.min(xmin,p.x);xmax=Math.max(xmax,p.x);ymin=Math.min(ymin,p.y);ymax=Math.max(ymax,p.y)});
function xy(p){return [((p.x-xmin)/(xmax-xmin)*.92+.04)*canvas.width*sx+tx,((p.y-ymin)/(ymax-ymin)*.92+.04)*canvas.height*sx+ty]}
function shown(p){return filter.value==='all'||p.domain===filter.value}
let palette=new Map();function draw(){canvas.width=Math.floor(canvas.clientWidth*devicePixelRatio);canvas.height=Math.floor(canvas.clientHeight*devicePixelRatio);const counts=new Map();points.filter(shown).forEach(p=>{let k=String(p[selector.value]);counts.set(k,(counts.get(k)||0)+1)});const keys=[...counts].sort((a,b)=>b[1]-a[1]).slice(0,20).map(v=>v[0]);palette=new Map(keys.map((k,i)=>[k,i<10?colours[i]:`hsl(${(i*137.508)%360},65%,45%)`]));const legend=document.querySelector('#legend');legend.replaceChildren();for(const k of keys){const span=document.createElement('span');span.style.color=palette.get(k);span.style.marginRight='12px';span.textContent=k+': '+counts.get(k);legend.append(span)}if(counts.size>20)legend.append(document.createTextNode('Other categories in grey'));ctx.clearRect(0,0,canvas.width,canvas.height);for(const p of points){if(!shown(p))continue;const [x,y]=xy(p);ctx.fillStyle=palette.get(String(p[selector.value]))||'#bbb';ctx.globalAlpha=p.domain==='external'?1:.45;ctx.beginPath();ctx.arc(x,y,(p.domain==='external'?3.2:1.1)*devicePixelRatio,0,Math.PI*2);ctx.fill();if(p.domain==='external'){ctx.strokeStyle='#111';ctx.lineWidth=.6*devicePixelRatio;ctx.stroke()}}ctx.globalAlpha=1}
selector.onchange=draw;filter.onchange=draw;document.querySelector('#reset').onclick=()=>{sx=1;tx=ty=0;draw()};window.onresize=draw;
canvas.onwheel=e=>{e.preventDefault();let rect=canvas.getBoundingClientRect(),x=(e.clientX-rect.left)*devicePixelRatio,y=(e.clientY-rect.top)*devicePixelRatio,f=Math.exp(-e.deltaY*.001);tx=x-(x-tx)*f;ty=y-(y-ty)*f;sx*=f;draw()};canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];moved=false;canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(Math.abs(dx)+Math.abs(dy)>2)moved=true;tx+=dx*devicePixelRatio;ty+=dy*devicePixelRatio;drag=[e.clientX,e.clientY];draw()};canvas.onpointerup=e=>{drag=null;if(moved)return;const rect=canvas.getBoundingClientRect(),x=(e.clientX-rect.left)*devicePixelRatio,y=(e.clientY-rect.top)*devicePixelRatio;let best=null,d=144*devicePixelRatio**2;for(const p of points){if(!shown(p))continue;const [a,b]=xy(p),dd=(a-x)**2+(b-y)**2;if(dd<d){d=dd;best=p}}if(best){document.querySelector('#info').textContent=JSON.stringify(best,null,2);document.querySelector('#audio').src=best.audio;const a=document.querySelector('#link');a.href=best.audio;a.textContent='Open selected audio'}};draw();</script></html>"""
    (REPORT / f"{method}_explorer.html").write_text(
        template.replace("METHOD", method).replace("DATA", data)
    )


def write_report():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    REPORT.mkdir(parents=True, exist_ok=True)
    for method in ("frozen", "partial"):
        path = ROOT / f"{method}_points.csv"
        if path.exists():
            with path.open() as stream:
                points = list(csv.DictReader(stream))
            for point in points:
                point["x"], point["y"] = float(point["x"]), float(point["y"])
                point["cluster"] = int(point["cluster"])
            explorer(method, points)
            static_figures(method, np.array([[p["x"], p["y"]] for p in points]), points)
    refresh_associations()
    sensitivity = json.loads((ROOT / "threshold_sensitivity.json").read_text())
    gate = json.loads((ROOT / "split_gate.json").read_text())
    audit = json.loads((ROOT / "audit_complete.json").read_text())
    figures = REPORT / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    thresholds = [s["threshold"] for s in sensitivity]
    axes[0].plot(
        thresholds,
        [s["speaker_clusters"] for s in sensitivity],
        "o-",
        label="Speaker graph components",
    )
    axes[0].plot(
        thresholds,
        [s["connected_recording_speaker_groups"] for s in sensitivity],
        "s-",
        label="Recording + speaker groups",
    )
    axes[0].set(
        xlabel="Cosine threshold", ylabel="Number of inferred groups", yscale="log"
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(
        thresholds,
        [
            100 * s["cuts_in_cross_split_clusters"] / audit["radio_cuts"]
            for s in sensitivity
        ],
        "o-",
        color="#d55e00",
    )
    axes[1].set(
        xlabel="Cosine threshold",
        ylabel="Cuts linked across existing splits (%)",
        ylim=(-2, 102),
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            figures / f"speaker_overlap.{extension}", dpi=350, bbox_inches="tight"
        )
    plt.close(fig)
    table = "\n".join(
        f"| {s['threshold']:.1f} | {s['speaker_clusters']:,} | {s['cross_original_split_clusters']:,} | {s['connected_recording_speaker_groups']:,} | {s['largest_group_cuts']:,} | {'Yes' if s['feasible'] else 'No'} |"
        for s in sensitivity
    )
    groups = "\n".join(
        f"| {code} | {count:,} |" for code, count in gate["groups_per_language"].items()
    )
    associations = []
    stability = []
    for method in ("frozen", "partial"):
        done = ROOT / f"{method}_geometry_complete.json"
        if done.exists():
            for value in json.loads(done.read_text())["associations"]:
                if value["attribute"] == "language":
                    nmi_display = (
                        f"{value['nmi']:.3f}"
                        if value["nmi"] is not None
                        else "undefined"
                    )
                    stability.append(
                        f"| {method} | {value['min_cluster_size']} | {value['clusters']} | {100 * value['noise_fraction']:.1f} | {nmi_display} |"
                    )
                if value["min_cluster_size"] == 50:
                    nmi = (
                        f"{value['nmi']:.3f}"
                        if value["nmi"] is not None
                        else "undefined"
                    )
                    ami = (
                        f"{value['ami']:.3f}"
                        if value["ami"] is not None
                        else "undefined"
                    )
                    associations.append(
                        f"| {method} | {value['attribute']} | {nmi} | {ami} | {100 * value['noise_fraction']:.1f} |"
                    )
    training = (
        "**Retraining stopped at the feasibility gate:** "
        + gate.get("reason", "")
        + ". No new classifiers were trained."
    )
    if gate["feasible"]:
        comparison = ROOT / "experiment_comparison.md"
        training = (
            comparison.read_text()
            if comparison.exists()
            else "Speaker split is feasible; training/evaluation is pending."
        )
    duration = "Paired duration diagnostic is pending."
    duration_figure = ""
    duration_links = ""
    duration_notes = ""
    if (ROOT / "duration_complete.json").exists() and not (
        ROOT / "duration_metrics.json"
    ).exists():
        duration = "No contiguous external speech regions were long enough for a paired 10-second window."
    baseline_models = json.loads((EXTERNAL / "report.json").read_text())["models"]
    baseline_text = "; ".join(
        f"{name}: {100 * baseline_models[key]['conditional']['accuracy']:.2f}% conditional accuracy"
        for name, key in (("frozen", "frozen15"), ("partial", "partial15"))
    )
    cohesion = ""
    if (ROOT / "speaker_cohesion.json").exists():
        result = json.loads((ROOT / "speaker_cohesion.json").read_text())
        if result["fraction_below_primary_threshold"] is not None:
            cohesion = f"- Of {result['cross_recording_sample_pairs']:,} randomly sampled cross-recording pairs within the largest speaker component, **{100 * result['fraction_below_primary_threshold']:.1f}% have cosine similarity below 0.5** (median {result['cosine_p10_p50_p90'][1]:.3f}). This measures weak component cohesion, without establishing the pairs' true identities.\n"
    if (ROOT / "duration_metrics.json").exists():
        values = json.loads((ROOT / "duration_metrics.json").read_text())
        lines = []
        for key, value in values.items():
            conditional, common = value["conditional"], value["common"]
            own_accuracy = (
                f"{100 * conditional['accuracy']:.2f}"
                if conditional["accuracy"] is not None
                else "undefined"
            )
            shared_accuracy = (
                f"{100 * common['accuracy']:.2f}"
                if common["accuracy"] is not None
                else "undefined"
            )
            lines.append(
                f"| {key} | {conditional['segments_total']} | {conditional['english_predictions']} | {own_accuracy} | {common['segments_scored']} | {shared_accuracy} |"
            )
        duration = (
            "| Model / crop | Paired windows | English exclusions | Conditional accuracy (%) | Common windows | Common accuracy (%) |\n|---|---:|---:|---:|---:|---:|\n"
            + "\n".join(lines)
        )
        fig, axis = plt.subplots(figsize=(6, 3.7), layout="constrained")
        for method, colour in (("frozen", "#0072B2"), ("partial", "#D55E00")):
            accuracy = [
                values[f"{method}_{seconds}s"]["common"]["accuracy"]
                for seconds in (3, 5, 10)
            ]
            display = [
                100 * value if value is not None else np.nan for value in accuracy
            ]
            axis.plot([3, 5, 10], display, "o-", color=colour, label=method.title())
            for seconds, value in zip((3, 5, 10), display):
                if np.isfinite(value):
                    axis.annotate(
                        f"{value:.1f}",
                        (seconds, value),
                        xytext=(0, 7 if method == "partial" else -13),
                        textcoords="offset points",
                        ha="center",
                        color=colour,
                        fontsize=8,
                    )
        common_count = values["frozen_10s"]["common"]["segments_scored"]
        subset = values["frozen_10s"]["conditional"]
        composition = ", ".join(
            f"{code}: {info['total']}" for code, info in subset["by_language"].items()
        )
        largest_source = max(info["total"] for info in subset["by_source"].values())
        source_fraction = largest_source / subset["segments_total"]
        duration_notes = (
            f"Eligible subset: **{composition}**, from {len(subset['by_source'])} source clips; "
            f"**{100 * source_fraction:.1f}%** of windows come from one source. "
            "This subset is not representative of the full external corpus; zero-count languages have no paired-duration evidence."
        )
        axis.set(
            xlabel="Crop length (seconds)",
            ylabel="Common-cohort accuracy (%)",
            xticks=[3, 5, 10],
            ylim=(0, 105),
            xlim=(2.5, 10.5),
            title=f"Matched external speech windows (n = {common_count})",
        )
        fig.text(
            0.5,
            -0.03,
            composition + "; labels remain provisional",
            ha="center",
            fontsize=8,
        )
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, loc="lower right")
        for extension in ("png", "pdf", "svg"):
            fig.savefig(
                figures / f"paired_duration.{extension}", dpi=350, bbox_inches="tight"
            )
        plt.close(fig)
        duration_figure = (
            "![Matched crop-length evaluation](figures/paired_duration.png)"
        )
        duration_links = "[Full duration metrics](../../artifacts/speaker_study/duration_metrics.json) · [Timestamped predictions](../../artifacts/speaker_study/duration_predictions.csv)."
    text = f"""# Pacific speaker overlap and generalisation

## Findings

- Audited **{audit["radio_cuts"]:,} radio cuts** using {audit["radio_windows"]:,} speaker windows. External speaker matches are diagnostics only.
- At primary cosine threshold 0.5: **{gate["speaker_clusters"]:,} inferred speaker graph components**, {gate["connected_recording_speaker_groups"]:,} combined recording/speaker groups; largest combined group contains {gate["largest_group_cuts"]:,} cuts.
- **{gate["cuts_in_cross_split_clusters"]:,} radio cuts** link to an inferred speaker component present in more than one original partition.
- These are automatic similarity groups, **not verified identities or proof of identity leakage**. False links and transitive chains can merge different people; missed links can split the same person.
- A collapsed graph across languages indicates that these permissive links cannot supply useful identity labels. This audit does not establish that the corpus contains too few genuine speakers. A validated matching threshold or a clustering method that limits transitive chaining is needed before claiming speaker-disjoint results.
{cohesion}

{training}

Existing external baseline: {baseline_text}. [Full existing evaluation](../youtube_validation/held_out_evaluation.md). No model or temperature was fitted on these clips.

## Speaker audit

- Speaker encoder: `speechbrain/spkrec-ecapa-voxceleb`, pinned in [provenance](../../artifacts/speaker_study/speaker_identity.json). The separate LID model was not used to infer speakers.
- Full cuts → Silero speech detection → contiguous nonoverlapping 3-second windows → normalized speaker embeddings. Native audio level; no added gain normalization.
- Exact blockwise cosine threshold graphs, evaluated globally across languages. Recording and duplicate links are added when constructing partitions. No full quadratic distance matrix is retained.
- A cut with several inferred speakers links all of them. {audit["unusable_cuts"]:,} radio/external cuts had no eligible 3-second speech window; these remain identity-uncertain. Unknown identities do not establish speaker independence.
- Thresholds are heuristic and have no manually measured false-match rate in these languages. A component is a group of potential matches, not a speaker-count estimate.
- Feasibility requires four independent groups and at least 50 cuts per language in each whole-group partition; targets are 80/5/5/10. Conservative grouping may prevent a valid split. We do not weaken thresholds after inspecting external performance.
- Speaker auditing and partition changes apply to the three Pacific languages. The original 12-language replay and FLEURS partitions are unchanged and are not speaker-audited in this study.

| Threshold | Speaker components | Components spanning original splits | Recording + speaker groups | Largest group cuts | Split feasible |
|---|---:|---:|---:|---:|---|
{table}

| Language | Independent groups at threshold 0.5 |
|---|---:|
{groups}

![Speaker grouping sensitivity](figures/speaker_overlap.png)

## LID embedding structure

- All Pacific radio cuts plus the fixed 428 external segments, separately for the base/frozen and existing partially fine-tuned encoder.
- UMAP: cosine metric, 30 neighbours, minimum distance 0.1, seed 42. Fit only on original radio training embeddings; transform all remaining points without language labels.
- Each encoder has an independent projection: axes, orientation and inter-cluster distances cannot be compared directly. A visible island is not proof of a true cluster.
- HDBSCAN is run on normalized embeddings reduced by training-fitted PCA50, **not on the 2D plot**. Minimum samples 10; minimum cluster sizes 25/50/100. Noise remains unassigned.
- Association scores below exclude HDBSCAN noise. Speaker associations use only radio cuts with a single assigned inferred speaker; unassigned and multiple-speaker cuts are excluded. Other attributes include external points. Scores are undefined if an attribute has fewer than two represented values. AMI adjusts for chance and is useful alongside NMI when recordings have many distinct IDs. Speaker labels inherit the automatic audit's uncertainty and the primary grouping's collapse makes speaker associations uninformative about genuine identities.
- Interactive views support colouring, domain filtering, pan/zoom, point metadata and local audio references. Open the HTML files locally with access to the original audio paths.

[Base/frozen interactive explorer](frozen_explorer.html) · [Partial interactive explorer](partial_explorer.html)

![Base LID embedding space](figures/frozen_umap.png)

![Partial LID embedding space](figures/partial_umap.png)

| Encoder | Attribute | NMI | AMI | Noise (%) |
|---|---|---:|---:|---:|
{chr(10).join(associations)}

Clustering sensitivity is reported for every requested minimum cluster size. Large changes in cluster counts, noise and language association mean that the chosen clustering settings should not be read as a count of natural groups.

| Encoder | Minimum cluster size | Clusters | Noise (%) | Language NMI |
|---|---:|---:|---:|---:|
{chr(10).join(stability)}

## Paired duration diagnostic

Matched 3-, 5- and 10-second center crops of the same contiguous external speech windows. The common cohort excludes a parent window if any of the six model/crop combinations predicts English. These clips remain evaluation-only, with provisional folder labels; changing crop length can also change the amount of code-switching in a window.

{duration}

{duration_notes}

{duration_figure}

{duration_links}

## Interpretation and next experiments

- Speaker-disjoint evaluation tests unseen-speaker robustness; changing the split alone does not guarantee better external performance. The number and balance of genuinely distinct speakers may be the limiting factor.
- Automatic groups cannot distinguish speaker identity from every channel, language or background effect. Review or labelled speaker pairs would be needed to validate the identities and tune a verification threshold.
- Preserve the external benchmark for evaluation. Its language labels remain provisional, English predictions are counted and excluded, and multiple cuts from one source are correlated. Current files may include overlapping extracts.
- Next priorities: manually labelled external speech; additional independent speakers and conversational styles; then separate noise/reverberation/codec augmentation ablations. Gain-only robustness was already tested and does not measure these effects.
- One training seed gives exploratory results without an estimate of training variance. Old-language regression remains part of every completed classifier comparison.
- These external clips have already informed experiment design. Treat them as development diagnostics and reserve fresh independent audio for a final generalisation claim.

## Artifacts

- [Per-cut inferred speaker labels](../../artifacts/speaker_study/speaker_labels.csv) · [Threshold sensitivity](../../artifacts/speaker_study/threshold_sensitivity.csv) · [External candidate matches](../../artifacts/speaker_study/external_speaker_matches.csv).
- [Base coordinates and metadata](../../artifacts/speaker_study/frozen_points.csv) · [Partial coordinates and metadata](../../artifacts/speaker_study/partial_points.csv).
- [Base cluster associations](../../artifacts/speaker_study/frozen_cluster_associations.csv) · [Partial cluster associations](../../artifacts/speaker_study/partial_cluster_associations.csv).
- Figures: PNG 350 dpi, PDF and SVG. Existing experiment results and source audio are preserved.
"""
    (REPORT / "speaker_generalisation.md").write_text(text)


def refresh_associations():
    """Exclude unassigned identities from speaker statistics, retaining all plot points."""
    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

    summaries = {}
    for method in ("frozen", "partial"):
        path = ROOT / f"{method}_points.csv"
        if not path.exists():
            continue
        with path.open() as stream:
            points = list(csv.DictReader(stream))
        results = []
        for minimum in (25, 50, 100):
            clusters = np.load(ROOT / f"{method}_clusters_{minimum}.npy")
            retained = clusters >= 0
            if len(clusters) != len(points):
                raise ValueError("Cluster metadata row count changed")
            for attribute in ("language", "speaker", "recording", "domain"):
                values = np.array([p[attribute] for p in points])
                mask = retained.copy()
                if attribute == "speaker":
                    mask &= np.array(
                        [
                            p["domain"] == "radio" and p["uncertain_speaker"] == "False"
                            for p in points
                        ]
                    )
                identifiable = mask.any() and len(set(values[mask])) >= 2
                results.append(
                    {
                        "min_cluster_size": minimum,
                        "attribute": attribute,
                        "n": int(mask.sum()),
                        "clusters": len(set(clusters[retained].tolist())),
                        "noise_fraction": float((~retained).mean()),
                        "nmi": float(
                            normalized_mutual_info_score(values[mask], clusters[mask])
                        )
                        if identifiable
                        else None,
                        "ami": float(
                            adjusted_mutual_info_score(values[mask], clusters[mask])
                        )
                        if identifiable
                        else None,
                    }
                )
        write_csv(ROOT / f"{method}_cluster_associations.csv", results)
        done = ROOT / f"{method}_geometry_complete.json"
        summary = json.loads(done.read_text())
        summary.update(
            associations=results,
            association_rule="noise excluded; speaker: radio, single assigned identity only; undefined for <2 attribute values",
        )
        write_json(done, summary)
        summaries[method] = summary
    if len(summaries) == 2:
        write_json(ROOT / "visualization_complete.json", summaries)
