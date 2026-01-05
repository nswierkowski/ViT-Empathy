import json
from pathlib import Path
from typing import Dict, List, Any

import torch
from tqdm import tqdm

from src.analysis.patching.patch_analysis import PatchExp
from src.utils.patching import patching_utils


class GroupComboEmbeddingPatchExp(PatchExp):
    def __init__(
        self,
        *args,
        groups_json_path: str | Path,
        file_key_in_batch: str = "file_name",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.groups_json_path = Path(groups_json_path)
        self.file_key_in_batch = file_key_in_batch

        if not self.groups_json_path.exists():
            raise FileNotFoundError(
                f"groups_json_path not found: {self.groups_json_path}"
            )

        self.group_db: Dict[str, Dict[str, List[int]]] = json.loads(
            self.groups_json_path.read_text()
        )

    def _patch_embeddings_tokens(
        self,
        emb_base: torch.Tensor,
        emb_source: torch.Tensor,
        patch_indices: List[int],
    ) -> torch.Tensor:
        emb_base = self._ensure_batch_hidden(emb_base).clone()
        emb_source = self._ensure_batch_hidden(emb_source)

        for p in patch_indices:
            tok = p + 1
            emb_base[:, tok, :] = emb_source[:, tok, :]
        return emb_base

    def _forward_from_embeddings_collect(
        self,
        emb: torch.Tensor,
    ) -> tuple[Dict[str, Any], Any, torch.Tensor]:
        _, layers, _ = self._get_vit_modules()
        num_layers = len(layers)
        final_layer_idx = num_layers - 1

        hidden = self._ensure_batch_hidden(emb)

        layer_probs: Dict[str, Any] = {}

        for layer_idx, layer in enumerate(layers):
            hidden = layer(hidden)
            if self.start_layer <= layer_idx <= self.last_layer:
                clf = self._load_lr_head(layer_idx)
                layer_probs[f"layer_{layer_idx}"] = self._lr_probs(clf, hidden)

        final_probs = self._final_head_probs(hidden, final_layer_idx)
        return layer_probs, final_probs, hidden

    def run(self):
        embeddings, layers, _ = self._get_vit_modules()
        num_layers = len(layers)
        assert num_layers > 0

        results = {
            "original": {},
            "corrupted": {},
            "patched": {},
            "emotion_map": self.emotion_map,
            "groups_json_path": str(self.groups_json_path),
        }

        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                file_key = patching_utils._get_sample_file_key(batch)
                groups = patching_utils._groups_for_file(file_key, self.group_db)
                group_names = list(groups.keys())
                combos = patching_utils._all_nonempty_combos(group_names)

                orig_img = batch["original_image"].to(self.device)
                corr_img = batch["corrupted_image"].to(self.device)

                orig_meta = self._unbatch_metadata(batch.get("original_metadata", {}))
                corr_meta = self._unbatch_metadata(batch.get("corrupted_metadata", {}))

                results["file_key"] = file_key
                results["groups"] = groups
                results["combo_names"] = ["+".join(c) for c in combos]
                results["original"]["metadata"] = orig_meta
                results["corrupted"]["metadata"] = corr_meta

                orig_emb = embeddings(orig_img)
                corr_emb = embeddings(corr_img)

                orig_layer_probs, orig_final_probs, _ = (
                    self._forward_from_embeddings_collect(orig_emb)
                )
                corr_layer_probs, corr_final_probs, _ = (
                    self._forward_from_embeddings_collect(corr_emb)
                )

                results["original"]["layer_probs"] = orig_layer_probs
                results["original"]["final_probs"] = orig_final_probs

                results["corrupted"]["layer_probs"] = corr_layer_probs
                results["corrupted"]["final_probs"] = corr_final_probs

                results["patched"] = {}

                n_tokens = orig_emb.shape[1]
                assert n_tokens == 197, (
                    f"Expected 197 tokens (CLS + 196 patches), got {n_tokens}"
                )
                num_patches = n_tokens - 1
                assert num_patches == 196

                for combo in tqdm(combos, desc="Group combos"):
                    combo_name = "+".join(combo)
                    patch_idxs = patching_utils._flatten_patch_indices(groups, combo)

                    # bounds check
                    assert len(patch_idxs) > 0
                    assert min(patch_idxs) >= 0 and max(patch_idxs) < num_patches, (
                        f"patch idx out of range in combo {combo_name}: {patch_idxs[:10]}"
                    )

                    emb_patched = self._patch_embeddings_tokens(
                        orig_emb, corr_emb, patch_idxs
                    )

                    layer_probs, final_probs, _ = self._forward_from_embeddings_collect(
                        emb_patched
                    )

                    results["patched"][combo_name] = {
                        "patched_groups": list(combo),
                        "patched_patch_indices": patch_idxs,
                        "layer_probs": layer_probs,
                        "final_probs": final_probs,
                    }

        out_path = self.path_to_save_results / "group_embedding_patching_results.pt"
        torch.save(results, out_path)
        return results
