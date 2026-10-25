"""Execute the deployment demo in a fresh Jupyter kernel with Python networking blocked.

Run with: uv run --offline --group notebook python scripts/check_offline_notebook.py
"""

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=PROJECT / "notebooks/offline_deployment_demo.ipynb",
    )
    parser.add_argument("--reverse-model-order", action="store_true")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Save executed example outputs to the notebook",
    )
    args = parser.parse_args()
    source = nbformat.read(args.notebook, as_version=4)
    nbformat.validate(source)
    original_ids = {cell.id for cell in source.cells}
    with tempfile.TemporaryDirectory(prefix="lid-notebook-check-") as temporary:
        guard = nbformat.v4.new_code_cell(
            "import os, socket, sys\n"
            f"assert sys.prefix == {sys.prefix!r}, sys.prefix\n"
            f"os.environ['HF_HOME'] = {str(Path(temporary) / 'empty-hf-cache')!r}\n"
            "_network_attempts = []\n"
            "def _no_network(self, address, *args, **kwargs):\n"
            "    _network_attempts.append(repr(address))\n"
            "    raise RuntimeError('Network access forbidden during notebook validation')\n"
            "socket.socket.connect = _no_network\n"
            "socket.socket.connect_ex = _no_network\n",
        )
        source.cells.insert(0, guard)
        if args.reverse_model_order:
            index = next(
                i
                for i, cell in enumerate(source.cells)
                if "load-models" in cell.metadata.get("tags", [])
            )
            source.cells.insert(
                index,
                nbformat.v4.new_code_cell(
                    "ENABLED_MODELS = list(reversed(ENABLED_MODELS))"
                ),
            )
        source.cells.append(
            nbformat.v4.new_code_cell(
                "assert _network_attempts == [], _network_attempts\n"
                "assert DEVICE == 'cpu'\n"
                "assert len(models) == 3\n"
                "assert all(len(codes) == 15 for codes in supported_codes.values())\n"
                "assert len(single_predictions) == len(models)\n"
                "assert all(len(predictions) == len(AUDIO_FILES) for predictions in batch_predictions.values())\n"
                "assert not any(name == 'lid' or name.startswith('lid.') for name in sys.modules)\n"
                "assert len({len(prepare_audio(path)) for path in AUDIO_FILES}) > 1\n"
                "assert 'custom' not in sys.modules\n"
                "print('OFFLINE CHECK PASSED: 3 models; 15 languages each; single and variable-length batch inference; no network or lid imports.')\n"
            )
        )
        client = NotebookClient(
            source,
            timeout=600,
            kernel_name="python3",
            resources={
                "metadata": {
                    "path": str(
                        PROJECT / "notebooks" if args.reverse_model_order else PROJECT
                    )
                }
            },
        )
        client.execute()
    output = "\n".join(
        item.get("text", "")
        for cell in source.cells
        if cell.cell_type == "code"
        for item in cell.get("outputs", [])
        if item.get("name") == "stdout"
    )
    print(output)
    if args.write:
        source.cells = [cell for cell in source.cells if cell.id in original_ids]
        source.metadata["offline_validation"] = {
            "executed_at_utc": datetime.now(UTC).isoformat(),
            "python_networking_blocked": True,
            "fresh_huggingface_cache": True,
            "device": "cpu",
            "models_tested": 3,
            "supported_languages_per_model": 15,
            "reverse_model_order": args.reverse_model_order,
        }
        nbformat.validate(source)
        nbformat.write(source, args.notebook)
        print(json.dumps({"executed_notebook_saved": str(args.notebook)}))


if __name__ == "__main__":
    main()
