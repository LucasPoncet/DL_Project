from __future__ import annotations

from typing import TypedDict, cast  
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ClassesData.WineDataModule import WineDataModule, WineDataset
from ClassesML.TabularMLP import TabularMLP
from ClassesML.TrainerTabular import TrainerClassifier
from Utils.Seed import set_seed


# ------------------------------------------------------------------ #
# 1)  Typage strict du dictionnaire hyperparamètres
# ------------------------------------------------------------------ #
class HParams(TypedDict):
    n_cont: int
    cat_cardinalities: list[int]   # rempli dynamiquement
    #emb_dims: list[int]            # rempli dynamiquement
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
    #"emb_dims": [],
    "hidden_layers_size": [64, 32],
    "activation": "relu",
    "batch_normalization": False,
    "dropout_rate": 0.1,
    "learning_rate": 2e-6,
    "max_epoch": 50,
    "batch_size": 64,
}
# ------------------------------------------------------------------ #

cont_cols = [
    "GDD",
    "TM_summer",
    "TX_summer",
    "temp_amp_summer",
    "hot_days",
    "rainy_days_summer",
    "rain_June",
    "rain_SepOct",
    "frost_days_Apr",
    "avg_TM_Apr",
]
cat_cols = ["cepages", "Winery"]


def main() -> None:
    set_seed(100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ─────────────────────────────────────────────────────────────── #
    dm = WineDataModule(
        train_parquet_path="data/vivino_wine_train_label.parquet",  # train data
        test_parquet_path="data/vivino_wine_test_label.parquet",          # test data
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        batch_size=int(hyper["batch_size"]),   # ← cast explicite
        label_col="label"
    )

    ds: WineDataset = cast(WineDataset, dm.train_ds.dataset)  # aide à Pylance
    cardinalities = ds.categorical_cardinalities()
    print(f"Original cardinalities: {cardinalities}")
    
    # Cap embedding dimensions to prevent NaN
    hyper["cat_cardinalities"] = cardinalities
    # ─────────────────────────────────────────────────────────────── #

    model = TabularMLP(cast(dict[str, object], hyper)).to(device)


    optimizer = Adam(model.parameters(), lr=float(hyper["learning_rate"]))  # cast
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.1)

    trainer = TrainerClassifier(cast(dict[str, object], hyper))
    trainer.set_model(model, device)
    trainer.set_scope(optimizer, scheduler)
    trainer.set_data(dm.train_dataloader(), dm.val_dataloader())
    trainer.run()

    import os
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hyperparameters': hyper,
        'epoch': hyper["max_epoch"],
    }, 'models/wine_model_best.pth')
    
    print(" Model saved to 'models/wine_model_best.pth'")


if __name__ == "__main__":
    main()
