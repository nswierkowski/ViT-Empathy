import json
from pathlib import Path
import joblib
from src.models.vit_backbones import load_expression_vit, load_imagenet_vit
from abc import ABC, abstractmethod


class PatchExp(ABC):
    def __init__(
        self,
        model_name: str,
        start_layer: int,
        last_layer: int,
        path_to_save_results,
        dataloader,
        preds_path: Path | None = None,
        emotion_map: dict = None,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.start_layer = start_layer
        self.last_layer = last_layer
        self.path_to_save_results = Path(path_to_save_results)
        self.dataloader = dataloader
        self.preds_path = preds_path
        self.device = device

        self.path_to_save_results.mkdir(parents=True, exist_ok=True)

        self.model = self._load_model().to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.emotion_map = (
            emotion_map
            if emotion_map is not None
            else {
                0: "neutrality",
                1: "happiness",
                2: "sadness",
                3: "anger",
                4: "disgust",
                5: "fear",
            }
        )
        self.streams = ["block_output", "attention_output", "mlp_activation"]

        self._save_metadata()

    def _load_model(self):
        if self.model_name in {
            "imagenet",
            "vit_imagenet",
            "google/vit-base-patch16-224-in21k",
        }:
            return load_imagenet_vit(device=self.device)
        if self.model_name in {
            "expression",
            "vit_expression",
            "trpakov/vit-face-expression",
        }:
            return load_expression_vit(device=self.device)
        raise ValueError(f"Unknown model_name: {self.model_name}")

    def _get_vit_modules(self):
        vit = self.model
        embeddings = vit.embeddings
        layers = vit.encoder.layer
        final_ln = vit.layernorm
        return embeddings, layers, final_ln

    def _patched_probe_next_layer(
        self,
        patched_layer_output,
        layer_idx: int,
    ):
        _, layers, _ = self._get_vit_modules()
        if layer_idx + 1 >= len(layers):
            return None

        clf_next = self._load_lr_head(layer_idx + 1)
        hidden_next = self._forward_one_layer(patched_layer_output, layer_idx + 1)
        return self._lr_probs(clf_next, hidden_next)

    def _ensure_batch_hidden(self, hidden):
        if hidden.dim() == 2:
            return hidden.unsqueeze(0)
        return hidden

    def _collect_hidden_states(self, pixel_values):
        embeddings, layers, _ = self._get_vit_modules()
        hidden = embeddings(pixel_values)
        hidden_states = [hidden]
        for layer in layers:
            out = layer(hidden)
            hidden = out
            hidden_states.append(hidden)
        return hidden_states

    def _load_lr_head(self, layer_idx):
        base_dir = self.preds_path
        lr_head_path = Path(base_dir) / "lr_layer_cls" / f"layer_{layer_idx}.joblib"
        if not lr_head_path.exists():
            raise FileNotFoundError(f"Missing LR head at: {lr_head_path}")
        return joblib.load(lr_head_path)["classifier"]

    def _final_cls_features(self, hidden):
        hidden = self._ensure_batch_hidden(hidden)
        return hidden[:, 0]

    def _final_head_probs(self, hidden_last, last_layer_idx: int):
        clf_last = self._load_lr_head(last_layer_idx)
        cls_feats = self._final_cls_features(hidden_last)
        return self._lr_probs_from_cls(clf_last, cls_feats)

    def _forward_one_layer(self, hidden, layer_idx: int):
        _, layers, _ = self._get_vit_modules()
        out = layers[layer_idx](self._ensure_batch_hidden(hidden))
        return out

    def _lr_probs_from_cls(self, clf, cls_feats_torch):
        x = cls_feats_torch.detach().cpu().numpy()
        return clf.predict_proba(x)

    def _lr_probs(self, clf, hidden):
        hidden = self._ensure_batch_hidden(hidden)
        cls = hidden[:, 0].detach().cpu().numpy()
        return clf.predict_proba(cls)

    def _unbatch_metadata(self, metadata):
        if isinstance(metadata, dict):
            return {
                key: value[0]
                if isinstance(value, (list, tuple)) and len(value) == 1
                else value
                for key, value in metadata.items()
            }
        return (
            metadata[0]
            if isinstance(metadata, (list, tuple)) and len(metadata) == 1
            else metadata
        )

    def _metadata_path(self):
        return self.path_to_save_results / "metadata.json"

    def _save_metadata(self):
        metadata = {
            "model_name": self.model_name,
            "start_layer": self.start_layer,
            "last_layer": self.last_layer,
        }
        with open(self._metadata_path(), "w") as f:
            json.dump(metadata, f, indent=2)

    def _compute_stream_tensors(self, layer_module, hidden_in):
        x = self._ensure_batch_hidden(hidden_in)

        x1 = layer_module.layernorm_before(x)
        attn_out = layer_module.attention(x1)
        attn_out = self._ensure_batch_hidden(attn_out)
        x_attn = x + attn_out

        x2 = layer_module.layernorm_after(x_attn)
        mlp_act = layer_module.intermediate(x2)
        mlp_act = self._ensure_batch_hidden(mlp_act)

        x_block = layer_module.output(mlp_act, x_attn)

        return {
            "attention_output": attn_out,
            "mlp_activation": mlp_act,
            "block_output": x_block,
        }

    @abstractmethod
    def run(self):
        pass
