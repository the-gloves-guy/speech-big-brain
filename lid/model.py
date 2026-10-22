"""SpeechBrain integration. Never backpropagate through ECAPA."""

import re

import torch
from huggingface_hub import HfApi, snapshot_download

from .languages import selected_indices

MODEL_ID = "speechbrain/lang-id-voxlingua107-ecapa"


def load_encoder(device, revision="main"):
    from speechbrain.inference.classifiers import EncoderClassifier

    resolved = (
        revision
        if re.fullmatch(r"[0-9a-f]{40}", revision)
        else HfApi(token=False).model_info(MODEL_ID, revision=revision).sha
    )
    snapshot = snapshot_download(
        MODEL_ID,
        revision=resolved,
        token=False,
        allow_patterns=[
            "hyperparams.yaml",
            "embedding_model.ckpt",
            "classifier.ckpt",
            "label_encoder.txt",
        ],
    )
    # SpeechBrain 1.0.3 does not forward revision to every pretrainer fetch.
    # Load both YAML and weights from the same immutable local snapshot.
    encoder = EncoderClassifier.from_hparams(
        source=snapshot,
        overrides={"pretrained_path": snapshot},
        savedir=f"artifacts/pretrained/{resolved}",
        run_opts={"device": device},
    )
    encoder.eval()
    encoder.requires_grad_(False)
    labels = [encoder.hparams.label_encoder.ind2lab[i] for i in range(107)]
    selected_indices(labels)  # Fail early on an unexpected model label vocabulary.
    # Capture true logits before softmax for filtering, calibration, and retraining.
    encoder.mods.classifier.softmax = torch.nn.Identity()
    return encoder, labels, resolved


def new_head(state, indices):
    from speechbrain.lobes.models.Xvector import Classifier

    head = Classifier(
        input_shape=[None, None, 256],
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=107,
    )
    head.softmax = torch.nn.Identity()
    head.load_state_dict(state)
    original = head.out.w
    reduced = torch.nn.Linear(original.in_features, len(indices), bias=True)
    with torch.no_grad():
        reduced.weight.copy_(original.weight[indices])
        reduced.bias.copy_(original.bias[indices])
    head.out = reduced
    return head


def cpu_state(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }
