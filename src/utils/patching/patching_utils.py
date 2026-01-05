import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any


def _get_sample_file_key(batch: Dict[str, Any]) -> str:
    md = batch["original_metadata"]["path"]
    vv = md[0]
    return Path(str(vv)).name


def _all_nonempty_combos(names: List[str]) -> List[Tuple[str, ...]]:
    combos = []
    for r in range(1, len(names) + 1):
        combos.extend(itertools.combinations(names, r))
    return combos


def _flatten_patch_indices(
    groups: Dict[str, List[int]], combo: Tuple[str, ...]
) -> List[int]:
    patch_idxs = []
    for g in combo:
        patch_idxs.extend(groups[g])
    patch_idxs = sorted(set(int(x) for x in patch_idxs))
    return patch_idxs


def _groups_for_file(file_key: str, group_db) -> Dict[str, List[int]]:
    if file_key not in group_db:
        raise KeyError(
            f"File key '{file_key}' not found in groups JSON. "
            f"Example keys: {list(group_db.keys())[:5]}"
        )
    groups = group_db[file_key]
    if not isinstance(groups, dict) or len(groups) == 0:
        raise ValueError(
            f"group_db[{file_key}] must be non-empty dict group->patch_list"
        )
    return groups
