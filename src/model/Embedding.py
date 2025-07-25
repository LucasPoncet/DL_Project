import pandas as pd
import torch
from typing import List, Dict, Tuple

def build_cat_mapping(
    data: pd.DataFrame | dict,
    cat_cols: List[str],
    existing: Dict[str, Dict[str, int]] | None = None,
) -> Tuple[Dict[str, Dict[str, int]], torch.Tensor, List[int]]:
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    mapping: Dict[str, Dict[str, int]] = existing or {}
    encoded_cols = []
    vocab_sizes  = []

    for col in cat_cols:
        if col not in mapping:
            mapping[col] = {}

        col_map = mapping[col]
        codes = []
        for val in df[col].astype(str):
            if val not in col_map:         
                col_map[val] = len(col_map)
            codes.append(col_map[val])
        encoded_cols.append(torch.tensor(codes, dtype=torch.long).unsqueeze(1))
        vocab_sizes.append(len(col_map))

    x_cat = torch.cat(encoded_cols, dim=1) if encoded_cols else torch.empty((len(df), 0), dtype=torch.long)
    return mapping, x_cat, vocab_sizes
