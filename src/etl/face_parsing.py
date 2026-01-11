import os
import json
from typing import Dict, List, Iterable, Union
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


FALLBACK_ID2LABEL = {
    0: "background",
    1: "skin",
    2: "l_brow",
    3: "r_brow",
    4: "l_eye",
    5: "r_eye",
    6: "eye_g",
    7: "l_ear",
    8: "r_ear",
    9: "ear_r",
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "neck_l",
    16: "cloth",
    17: "hair",
    18: "hat",
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _get_filename(path: str) -> str:
    return os.path.basename(path)


@torch.no_grad()
def _predict_masks(
    model,
    processor,
    images_bchw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    inputs = {"pixel_values": images_bchw.to(device)}
    outputs = model(**inputs)
    logits = outputs.logits

    if logits.shape[-2:] != (224, 224):
        logits = F.interpolate(
            logits, size=(224, 224), mode="bilinear", align_corners=False
        )

    masks = torch.argmax(logits, dim=1).to(torch.long)  # [B, 224, 224]
    return masks


def _build_palette(num_classes: int) -> list:
    palette = []
    for i in range(256):
        r = (i * 37) % 256
        g = (i * 17) % 256
        b = (i * 29) % 256
        palette.extend([r, g, b])
    return palette[: 256 * 3]


def _save_mask_image_color(
    mask_hw: torch.Tensor, save_path: str, num_classes: int
) -> None:
    mask_np = (
        mask_hw.detach().cpu().to(torch.uint8).numpy()
    )  # [H,W], values 0..num_classes-1
    img = Image.fromarray(mask_np, mode="P")
    img.putpalette(_build_palette(num_classes))
    img.save(save_path)


IdGroup = Union[int, Sequence[int]]


def _normalize_groups(ids_to_save: Iterable[IdGroup]) -> List[Tuple[int, ...]]:
    groups: List[Tuple[int, ...]] = []
    for item in ids_to_save:
        if isinstance(item, int):
            groups.append((int(item),))
        elif isinstance(item, (list, tuple, set)):
            groups.append(tuple(int(x) for x in item))
        else:
            raise TypeError(
                "ids_to_save must contain ints or sequences of ints, "
                f"got element {item} of type {type(item)}"
            )
    return groups


def _group_name(group: Tuple[int, ...], id2label: Dict[int, str]) -> str:
    labels = [id2label.get(i, f"class_{i}") for i in group]
    s = set(labels)

    if s == {"l_eye", "r_eye"}:
        return "eyes"
    if s == {"l_brow", "r_brow"}:
        return "brows"
    if s == {"u_lip", "l_lip", "mouth"}:
        return "lips"
    
    if len(group) == len(FALLBACK_ID2LABEL):
        return "full_face"

    stripped = []
    for lab in labels:
        if lab.startswith("l_"):
            stripped.append(lab[2:])
        elif lab.startswith("r_"):
            stripped.append(lab[2:])
        else:
            stripped.append(lab)

    if len(set(stripped)) == 1 and len(group) > 1:
        return stripped[0]

    return "_".join(labels)


def _mask_to_group_patch_membership(
    mask_hw: torch.Tensor,
    groups: List[Tuple[int, ...]],
    patch_grid: int = 14,
    patch_size: int = 16,
) -> Dict[Tuple[int, ...], List[int]]:
    assert mask_hw.shape == (224, 224), (
        f"Expected (224,224), got {tuple(mask_hw.shape)}"
    )

    m = mask_hw.view(patch_grid, patch_size, patch_grid, patch_size).permute(
        0, 2, 1, 3
    ) 

    out: Dict[Tuple[int, ...], List[int]] = {}
    for group in groups:
        present = torch.zeros(
            (patch_grid, patch_grid), dtype=torch.bool, device=m.device
        )
        for cid in group:
            present |= (m == int(cid)).any(dim=(2, 3))
        idxs = present.flatten().nonzero(as_tuple=False).flatten().tolist()
        if idxs:
            out[group] = idxs
    return out


def run(
    dataloader,
    path_to_save_json: str,
    save_mask: bool,
    ids_to_save: Union[List[int], torch.Tensor, set],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "jonathandinu/face-parsing"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", None) or FALLBACK_ID2LABEL

    groups = _normalize_groups(ids_to_save)

    if isinstance(ids_to_save, torch.Tensor):
        ids_to_save = ids_to_save.detach().cpu().tolist()
    json_dir = os.path.dirname(os.path.abspath(path_to_save_json)) or "."
    mask_dir = os.path.join(json_dir, "masks")
    if save_mask:
        _ensure_dir(mask_dir)

    result: Dict[str, Dict[str, List[int]]] = {}

    for batch in tqdm(dataloader, desc="Face parsing + patch mapping"):
        images = batch["image"]
        paths = batch["path"]

        masks_bhw = _predict_masks(model, processor, images, device)

        for i in range(masks_bhw.shape[0]):
            filename = (
                _get_filename(paths[i])
                if isinstance(paths, (list, tuple))
                else _get_filename(str(paths[i]))
            )
            mask_hw = masks_bhw[i]

            if save_mask:
                mask_path = os.path.join(
                    mask_dir, os.path.splitext(filename)[0] + "_mask.png"
                )
                _save_mask_image_color(
                    mask_hw, mask_path, num_classes=model.config.num_labels
                )

            group_to_patches = _mask_to_group_patch_membership(
                mask_hw, groups, patch_grid=14, patch_size=16
            )

            per_image: Dict[str, List[int]] = {}
            for group, patch_list in group_to_patches.items():
                key = _group_name(group, id2label)
                per_image[key] = patch_list

            result[filename] = per_image

    _ensure_dir(json_dir)
    with open(path_to_save_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
