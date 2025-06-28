# src/model/main_tabular.py
from __future__ import annotations

from typing import TypedDict, cast  
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ClassesData.WineDataModule import WineDataModule, WineDataset
from ClassesML.TabularMLP import TabularMLP
from ClassesML.TrainerTabular import TrainerTabular
from Utils.Seed import set_seed


# ------------------------------------------------------------------ #
# 1)  Typage strict du dictionnaire hyperparamètres
# ------------------------------------------------------------------ #
class HParams(TypedDict):
    n_cont: int
    cat_cardinalities: list[int]   # rempli dynamiquement
    emb_dims: list[int]            # rempli dynamiquement
    hidden_layers_size: list[int]
    activation: str
    batch_normalization: bool
    dropout_rate: float
    learning_rate: float
    max_epoch: int
    batch_size: int


hyper: HParams = {
    "n_cont": 10,
    "cat_cardinalities": [],       # placeholders
    "emb_dims": [],
    "hidden_layers_size": [128, 128, 128],
    "activation": "relu",
    "batch_normalization": False,
    "dropout_rate": 0.2,
    "learning_rate": 1e-3,
    "max_epoch": 50,
    "batch_size": 256,
}
# ------------------------------------------------------------------ #

cont_cols = [
    "GDD",
    "TM_SUMMER",
    "TX_SUMMER",
    "temp_amp_summer",
    "hot_days",
    "rainy_days_summer",
    "rain_June",
    "rain_SepOct",
    "frost_days_Apr",
    "avg_TM_Apr",
]
cat_cols = ["cepage", "winery"]


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─────────────────────────────────────────────────────────────── #
    dm = WineDataModule(
        parquet_path="data/weather_csv/wines.parquet",
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        batch_size=int(hyper["batch_size"]),   # ← cast explicite
    )

    ds: WineDataset = cast(WineDataset, dm.train_ds.dataset)  # aide à Pylance
    hyper["cat_cardinalities"] = ds.categorical_cardinalities()
    hyper["emb_dims"] = ds.default_embedding_dims()
    # ─────────────────────────────────────────────────────────────── #

    model = TabularMLP(cast(dict[str, object], hyper)).to(device)

    optimizer = Adam(model.parameters(), lr=float(hyper["learning_rate"]))  # cast
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.1)

    trainer = TrainerTabular(cast(dict[str, object], hyper))
    trainer.set_model(model, device)
    trainer.set_scope(optimizer, scheduler)
    trainer.set_data(dm.train_dataloader(), dm.val_dataloader())
    trainer.run()


if __name__ == "__main__":
    main()
