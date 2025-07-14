from typing import List, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset

class WineSeqDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 seq_len: int,
                 num_cols: List[str],
                 cat_cols: List[str],
                 label_col: str,
                 static_cols: Optional[List[str]] = None,
                 year_col: str = 'year',
                 pad_mode: str = 'zeros'):
        super().__init__()
        self.seq_len     = seq_len
        self.num_cols    = num_cols
        self.cat_cols    = cat_cols
        self.label_col   = label_col
        self.static_cols = static_cols or []
        self.year_col    = year_col
        self.pad_mode    = pad_mode

        # Sort by region then year for chronological order
        df_sorted = df.sort_values([cat_cols[0], year_col])

        self.samples = []
        for region, g in df_sorted.groupby(cat_cols[0]):
            g = g.reset_index(drop=True)
            for idx in range(len(g)):
                start = max(0, idx - seq_len + 1)
                window = g.iloc[start: idx + 1]
                if len(window) < seq_len:
                    if pad_mode == 'skip':
                        continue
                    if pad_mode == 'zeros':
                        pad_rows = [window.iloc[0].copy() for _ in range(seq_len - len(window))]
                        window = pd.concat([pd.DataFrame(pad_rows), window], ignore_index=True)
                # ENFORCE window length
                if len(window) == seq_len and not pd.isna(window.iloc[-1][label_col]):
                    self.samples.append(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window = self.samples[idx]
        window = window.sort_values(self.year_col)

        # Sequence features
        x_seq = torch.tensor(window[self.num_cols].values, dtype=torch.float32)

        # Categorical features (last row)
        last_row = window.iloc[-1]
        cat_series = pd.Series(last_row[self.cat_cols]).infer_objects(copy=False)
        cat_vals = cat_series.fillna(-1).astype(int).values
        x_cat = torch.tensor(cat_vals, dtype=torch.long)

        # Static numeric features (last row)
        if self.static_cols:
            stat_series = pd.Series(last_row[self.static_cols]).infer_objects(copy=False)
            stat_vals = stat_series.fillna(0).astype(float).values
            x_stat = torch.tensor(stat_vals, dtype=torch.float32)
        else:
            x_stat = torch.empty(0, dtype=torch.float32)


        # Label (last row)
        label_val = last_row[self.label_col]
        if pd.isna(label_val):
            label_val = -1
        y = torch.tensor(int(label_val), dtype=torch.long)

        return x_seq, x_cat, x_stat, y