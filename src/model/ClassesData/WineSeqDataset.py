# ---------------------------------------------------------------------------
# Dataset utilitaire
# Génère pour chaque (région × millésime) un échantillon contenant :
#   • x_seq   : séquence météo (T, F_seq)
#   • x_cat   : indices catégoriels [region, station, cépage]
#   • x_stat  : attributs statiques numériques (ex. price)
#   • y       : label binaire (0/1)
# ---------------------------------------------------------------------------

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
        """Paramètres
        df         : DataFrame au format long (une ligne = 1 millésime)
        seq_len    : taille de la fenêtre temporelle (T)
        num_cols   : colonnes météo numériques (F_seq)
        cat_cols   : colonnes catégorielles déjà encodées en int64 (ordre: region, station, cépage)
        label_col  : colonne cible binaire
        static_cols: colonnes statiques numériques (ex. ['price'])
        year_col   : colonne d'ordre chronologique (par défaut 'year')
        pad_mode   : 'zeros' (padding) ou 'skip' (on ignore si pas assez d'historique)
        """
        super().__init__()
        self.seq_len     = seq_len
        self.num_cols    = num_cols
        self.cat_cols    = cat_cols
        self.label_col   = label_col
        self.static_cols = static_cols or []
        self.year_col    = year_col
        self.pad_mode    = pad_mode

        # Trie par région puis année pour assurer l'ordre chronologique
        df_sorted = df.sort_values([cat_cols[0], year_col])

        self.samples = []  # liste de mini‑dataframes (fenêtres)
        for region, g in df_sorted.groupby(cat_cols[0]):
            g = g.reset_index(drop=True)
            for idx in range(len(g)):
                start = max(0, idx - seq_len + 1)
                window = g.iloc[start: idx + 1]
                if len(window) < seq_len:
                    if pad_mode == 'skip':
                        continue
                    if pad_mode == 'zeros':
                        # Padding par copies de la première ligne puis remise en ordre
                        pad_rows = [window.iloc[0]] * (seq_len - len(window))
                        window = pd.concat(pad_rows + [window])
                self.samples.append(window)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        window = self.samples[idx]
        window = window.sort_values(self.year_col)

        # Séquence météo (T, F_seq)
        x_seq = torch.tensor(window[self.num_cols].values, dtype=torch.float32)

        # Catégorielles → dernière ligne (millésime cible)
        last_row = window.iloc[-1]
        cat_vals = pd.Series(last_row[self.cat_cols]).fillna(-1).astype(int).values
        x_cat = torch.tensor(cat_vals, dtype=torch.long)

        # Statique numérique (ex. price)
        if self.static_cols:
            x_stat = torch.tensor(last_row[self.static_cols].values.astype(float), dtype=torch.float32)
        else:
            x_stat = torch.empty(0, dtype=torch.float32)

        # Label binaire
        y = torch.tensor(int(last_row[self.label_col]), dtype=torch.long)

        return x_seq, x_cat, x_stat, y
