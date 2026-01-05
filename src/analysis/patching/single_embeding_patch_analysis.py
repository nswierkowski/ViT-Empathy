import torch
from tqdm import tqdm
from src.analysis.patching.patch_analysis import PatchExp


class EmbeddingPatchExp(PatchExp):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _forward_from_embeddings_collect(self, x_embed):
        embeddings, layers, final_ln = self._get_vit_modules()
        x = self._ensure_batch_hidden(x_embed)

        num_layers = len(layers)
        assert x.shape[0] == 1 and x.shape[1] == 197 and x.shape[2] == 768, (
            f"bad embed shape: {x.shape}"
        )

        start = max(0, self.start_layer)
        end = min(num_layers - 1, self.last_layer)

        layer_probs = {}

        for i in range(num_layers):
            x = layers[i](x)
            x = self._ensure_batch_hidden(x)
            assert x.shape == (1, 197, 768), f"bad hidden after layer {i}: {x.shape}"

            if start <= i <= end:
                clf = self._load_lr_head(i)
                layer_probs[f"layer_{i}"] = self._lr_probs(clf, x)

        clf_last = self._load_lr_head(num_layers - 1)
        final_probs = self._lr_probs(clf_last, x)

        return layer_probs, final_probs

    def _patch_embeddings(self, orig_emb, corr_emb, token_indices):
        orig_emb = self._ensure_batch_hidden(orig_emb)
        corr_emb = self._ensure_batch_hidden(corr_emb)
        assert orig_emb.shape == corr_emb.shape == (1, 197, 768)

        patched = orig_emb.clone()
        for ti in token_indices:
            if (ti == 0) and (not self.patch_cls):
                continue
            patched[:, ti, :] = corr_emb[:, ti, :]
        return patched

    def run(self):
        embeddings, layers, final_ln = self._get_vit_modules()

        results = {
            "original": {},
            "corrupted": {},
            "patched": {},
            "emotion_map": self.emotion_map,
        }

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Batches"):
                orig_img = batch["original_image"].to(self.device)
                corr_img = batch["corrupted_image"].to(self.device)

                orig_meta = self._unbatch_metadata(batch["original_metadata"])
                corr_meta = self._unbatch_metadata(batch["corrupted_metadata"])

                results["original"]["metadata"] = orig_meta
                results["corrupted"]["metadata"] = corr_meta

                orig_emb = embeddings(orig_img)
                corr_emb = embeddings(corr_img)
                orig_emb = self._ensure_batch_hidden(orig_emb)
                corr_emb = self._ensure_batch_hidden(corr_emb)

                assert orig_emb.shape == (1, 197, 768)
                assert corr_emb.shape == (1, 197, 768)

                orig_layer_probs, orig_final_probs = (
                    self._forward_from_embeddings_collect(orig_emb)
                )
                corr_layer_probs, corr_final_probs = (
                    self._forward_from_embeddings_collect(corr_emb)
                )

                results["original"]["layer_probs"] = orig_layer_probs
                results["original"]["final_probs"] = orig_final_probs

                results["corrupted"]["layer_probs"] = corr_layer_probs
                results["corrupted"]["final_probs"] = corr_final_probs

                num_patches = 196
                results["patched"] = {}

                for patch_idx in tqdm(range(num_patches), desc="Patches"):
                    token_idx = patch_idx + 1
                    patched_emb = self._patch_embeddings(
                        orig_emb, corr_emb, [token_idx]
                    )

                    layer_probs, final_probs = self._forward_from_embeddings_collect(
                        patched_emb
                    )

                    results["patched"][f"patch_{patch_idx}"] = {
                        "token_index": token_idx,
                        "layer_probs": layer_probs,
                        "final_probs": final_probs,
                    }

        out_path = self.path_to_save_results / "embedding_patching_results.pt"
        torch.save(results, out_path)
        print("Saved:", out_path)
        return results
