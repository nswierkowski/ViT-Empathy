import json
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from src.analysis.patching.patch_analysis import PatchExp
from src.utils.patching import patching_utils


class GroupComboPatchExp(PatchExp):
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

    def _apply_stream_patch_multi(
        self,
        layer_module,
        hidden_in,
        stream: str,
        patch_indices: List[int],
        source_tensor,
    ):
        x = self._ensure_batch_hidden(hidden_in)
        source_tensor = self._ensure_batch_hidden(source_tensor)

        x1 = layer_module.layernorm_before(x)
        attn_out = layer_module.attention(x1)
        attn_out = self._ensure_batch_hidden(attn_out)

        if stream == "attention_output":
            attn_out = attn_out.clone()
            for p in patch_indices:
                token_index = p + 1
                attn_out[:, token_index, :] = source_tensor[:, token_index, :]
        x = x + attn_out

        x2 = layer_module.layernorm_after(x)
        mlp_act = layer_module.intermediate(x2)
        mlp_act = self._ensure_batch_hidden(mlp_act)

        if stream == "mlp_activation":
            mlp_act = mlp_act.clone()
            for p in patch_indices:
                token_index = p + 1
                mlp_act[:, token_index, :] = source_tensor[:, token_index, :]

        mlp_out = layer_module.output(mlp_act, x)
        x = self._ensure_batch_hidden(mlp_out)

        if stream == "block_output":
            x = x.clone()
            for p in patch_indices:
                token_index = p + 1
                x[:, token_index, :] = source_tensor[:, token_index, :]

        return x

    def run(self):
        embeddings, layers, final_ln = self._get_vit_modules()
        num_layers = len(layers)
        final_layer_idx = num_layers - 1

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

                results["original"]["metadata"] = orig_meta
                results["corrupted"]["metadata"] = corr_meta
                results["file_key"] = file_key
                results["groups"] = groups
                results["combo_names"] = ["+".join(c) for c in combos]

                orig_states = self._collect_hidden_states(orig_img)
                corr_states = self._collect_hidden_states(corr_img)

                orig_final_probs = self._final_head_probs(
                    orig_states[-1], final_layer_idx
                )
                corr_final_probs = self._final_head_probs(
                    corr_states[-1], final_layer_idx
                )
                results["original"]["final_probs"] = orig_final_probs
                results["corrupted"]["final_probs"] = corr_final_probs

                results["original"].setdefault("layer_probs", {})
                results["corrupted"].setdefault("layer_probs", {})

                for layer_idx in range(self.start_layer, self.last_layer + 1):
                    layer_key = f"layer_{layer_idx}"
                    clf = self._load_lr_head(layer_idx)

                    orig_hidden = self._ensure_batch_hidden(orig_states[layer_idx + 1])
                    corr_hidden = self._ensure_batch_hidden(corr_states[layer_idx + 1])

                    results["original"]["layer_probs"][layer_key] = self._lr_probs(
                        clf, orig_hidden
                    )
                    results["corrupted"]["layer_probs"][layer_key] = self._lr_probs(
                        clf, corr_hidden
                    )
                    results["patched"][layer_key] = {}

                    orig_in = self._ensure_batch_hidden(orig_states[layer_idx])
                    corr_in = self._ensure_batch_hidden(corr_states[layer_idx])

                    corr_streams = self._compute_stream_tensors(
                        layers[layer_idx], corr_in
                    )

                    num_patches = orig_in.shape[1] - 1
                    assert num_patches == 196, (
                        f"Expected 196 patches, got {num_patches}"
                    )

                    for stream in tqdm(self.streams, desc=f"Layer {layer_idx} streams"):
                        results["patched"][layer_key][stream] = {}

                        for combo in tqdm(combos):
                            combo_name = "+".join(combo)
                            patch_idxs = patching_utils._flatten_patch_indices(
                                groups, combo
                            )

                            assert len(patch_idxs) > 0
                            assert (
                                min(patch_idxs) >= 0 and max(patch_idxs) < num_patches
                            ), (
                                f"Patch idx out of range in combo {combo_name}: {patch_idxs[:10]}..."
                            )

                            patched_L = self._apply_stream_patch_multi(
                                layer_module=layers[layer_idx],
                                hidden_in=orig_in,
                                stream=stream,
                                patch_indices=patch_idxs,
                                source_tensor=corr_streams[stream],
                            )

                            hidden = patched_L
                            for k in range(layer_idx + 1, num_layers):
                                hidden = layers[k](hidden)

                            patched_final_probs = self._final_head_probs(
                                hidden, final_layer_idx
                            )

                            probe_probs = (
                                self._patched_probe_next_layer(patched_L, layer_idx)
                                if (layer_idx + 1 < num_layers)
                                else None
                            )

                            results["patched"][layer_key][stream][combo_name] = {
                                "patched_groups": list(combo),
                                "patched_patch_indices": patch_idxs,
                                "final_probs": patched_final_probs,
                                "probe_next_layer_probs": probe_probs,
                            }

        out_path = self.path_to_save_results / "group_patching_results.pt"
        torch.save(results, out_path)
        return results
