import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ClassesData.WineDataModule import DatasetLoader
from ClassesData.WineSeqDataset import WineSeqDataset
from ClassesML.RNN import WineRNN
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerTabular import TrainerClassifier
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 1. Load parquet splits via DatasetLoader -------------------
train_path = "data/vivino_wine_train_label.parquet"
test_path  = "data/vivino_wine_test_label.parquet"

num_cols = [
    "GDD","TM_summer","TX_summer","temp_amp_summer","hot_days",
    "rainy_days_summer","rain_June","rain_SepOct",
    "frost_days_Apr","avg_TM_Apr"
]
static_cols = ["price"]
cat_cols = ["region", "station", "cepages"]

loader = DatasetLoader(
    train_path  = train_path,
    test_path   = test_path,
    target_col  = "label",
    num_cols    = num_cols,
    cat_cols    = cat_cols,
    valid_frac  = 0.2,
    dtype       = torch.float32,
)

train_df, valid_df, _, num_cols, cat_cols, map_ = loader.load_sequence_dataframe()

# ---------- 2. Clean categorical and static columns --------------------
for col in cat_cols:
    train_df[col] = train_df[col].fillna(-1).astype(int)
    valid_df[col] = valid_df[col].fillna(-1).astype(int)

for col in static_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
    valid_df[col] = pd.to_numeric(valid_df[col], errors='coerce').fillna(0)

for df in [train_df, valid_df]:
    df['label'] = df['label'].fillna(-1).astype(int)

print("NaNs in train_df['label']:", train_df['label'].isna().sum())
print("NaNs in valid_df['label']:", valid_df['label'].isna().sum())

# ---------- 3. Prepare sequence datasets and loaders -------------------
seq_len = 5
train_ds = WineSeqDataset(train_df, seq_len, num_cols, cat_cols,
                          label_col='label', static_cols=static_cols)
valid_ds = WineSeqDataset(valid_df, seq_len, num_cols, cat_cols,
                          label_col='label', static_cols=static_cols)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=512, shuffle=False)

# ---------- 4. Hyperparameters and model -------------------------------
embed_sizes = {
    'region':  (len(map_['region']), 8),
    'station': (len(map_['station']),16),
    'cepages': (len(map_['cepages']), 8),
}
hp = dict(
    seq_len             = seq_len,
    num_seq_features    = len(num_cols),
    static_num_features = len(static_cols),
    embedding_sizes     = embed_sizes,
    rnn_hidden_size     = [64],
    num_layers          = 1,
    dropout_rate        = 0.2,
    bidirectional       = False,
    output_dim          = 2,
    learning_rate       = 1e-3,
    max_epoch           = 50,
)

model = WineRNN(hp).to(device)
scope = ScopeClassifier(model, hp, steps_per_epoch=len(train_loader))

# ---------- 5. Balanced loss -------------------------------------------
# Collect all labels from train_ds
all_labels = []
for _, _, _, y in train_loader:
    all_labels.append(y)
y_train = torch.cat(all_labels)
cnt = Counter(y_train.cpu().numpy())
total = len(y_train)
weights = [total / cnt[cls] for cls in [0, 1]]
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
scope.criterion = criterion

# ---------- 6. Trainer workflow ----------------------------------------
trainer = TrainerClassifier(hyperparameter=hp)
trainer.set_model(model=model, device=device)
trainer.set_scope(scope=scope)
trainer.set_data_loaders(train_loader, valid_loader)

train_acc_hist, valid_acc_hist = trainer.run()

# ---------- 7. Plot accuracy curves ------------------------------------
import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_acc_hist,  label="Train accuracy")
plt.plot(valid_acc_hist,  label="Valid accuracy")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()