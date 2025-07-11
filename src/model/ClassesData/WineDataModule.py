import os, pandas as pd, torch
import torch
from torch.utils.data import TensorDataset
from category_encoders import OneHotEncoder            # pip install category_encoders

class DatasetLoader:
    def __init__(
        self,
        root: str,
        target_col: str = "label",
        num_cols: list[str] | None = None,
        onehot_cols: list[str] = ["region", "station", "cepage"],  # ← add cepage
        valid_frac: float = 0.2,
        dtype: torch.dtype = torch.float32,
    ):
        self.root, self.target_col = root, target_col
        self.num_cols, self.onehot_cols = num_cols, onehot_cols
        self.valid_frac, self.dtype = valid_frac, dtype
        self.oh: OneHotEncoder | None = None
        self.onehot_dim: int = 0

    # -------------------------------------------------------------- #
    def load_tabular_data(self):
        train_valid = pd.read_parquet(os.path.join(self.root, "train_valid.parquet"))
        test        = pd.read_parquet(os.path.join(self.root, "test.parquet"))

        # choose numeric cols automatically if None
        if self.num_cols is None:
            excluded = set(self.onehot_cols + [self.target_col])
            self.num_cols = [c for c in train_valid.columns if c not in excluded]

        # train/valid split
        if "split" in train_valid.columns:
            train_df = train_valid[train_valid["split"] == "train"].copy()
            valid_df = train_valid[train_valid["split"] == "valid"].copy()
        else:
            train_df = train_valid.sample(frac=1 - self.valid_frac, random_state=42)
            valid_df = train_valid.drop(train_df.index)

        # ── fit OneHotEncoder on train split ─────────────────────────
        self.oh = OneHotEncoder(cols=self.onehot_cols, handle_unknown="ignore", use_cat_names=True)
        self.oh.fit(train_df[self.onehot_cols])
        self.onehot_dim = len(self.oh.get_feature_names_out())

        # helper to transform a dataframe → tensors
        def df_to_ds(df: pd.DataFrame) -> TensorDataset:
            x_num    = torch.tensor(df[self.num_cols].values,             dtype=self.dtype)
            x_1hot   = torch.tensor(self.oh.transform(df[self.onehot_cols]).values, # type: ignore
                                   dtype=self.dtype)
            y        = torch.tensor(df[self.target_col].values, dtype=torch.long)
            return TensorDataset(x_num, x_1hot, y)

        train_ds = df_to_ds(train_df)
        valid_ds = df_to_ds(valid_df)
        test_ds  = df_to_ds(test)

        meta = dict(
            num_dim   = len(self.num_cols),
            onehot_dim= self.onehot_dim,        # region + station + cepage dummies
        )
        n_classes = int(train_valid[self.target_col].nunique())
        return train_ds, valid_ds, test_ds, meta, n_classes
