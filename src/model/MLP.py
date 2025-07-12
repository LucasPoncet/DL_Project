# ------------------------------------------------------------
#  COMPLETE TRAINING SCRIPT  –  binary wine-quality classifier
# ------------------------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import lightgbm as lgb

from ClassesData.WineDataModule import DatasetLoader
from ClassesML.TabularMLP       import TabularMLP
from ClassesML.Scope            import ScopeClassifier
from ClassesML.TrainerTabular   import TrainerClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1. Load parquet splits via DatasetLoader -------------------
train_path = "data/vivino_wine_train_label.parquet"
test_path  = "data/vivino_wine_test_label.parquet"

num_cols = [
    "GDD","TM_summer","TX_summer","temp_amp_summer","hot_days",
    "rainy_days_summer","rain_June","rain_SepOct",
    "frost_days_Apr","avg_TM_Apr", "price"
]
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

train_ds, valid_ds, test_ds, onehot_mapping, _ = loader.load_tabular_data()

# ---------- 2. Clean numerical data (nan / inf) ------------------------
for ds in (train_ds, valid_ds):
    x_num = torch.nan_to_num(ds.tensors[0], nan=0.0, posinf=0.0, neginf=0.0)
    ds.tensors = (x_num, *ds.tensors[1:])

# ---------- 3. Prepare data splits -------------------------------------
x_num_train, x_cat_train, y_train = train_ds.tensors
x_num_valid, x_cat_valid, y_valid = valid_ds.tensors

print("Label counts:", np.bincount(y_train.numpy()))

# Count unique values for embeddings
num_regions  = len(onehot_mapping['region'])
num_stations = len(onehot_mapping['station'])
num_cepages  = len(onehot_mapping['cepages'])

# ---------- 4. Optional shape check ------------------------------------
train_loader = DataLoader(TensorDataset(x_num_train, x_cat_train, y_train), batch_size=512, shuffle=True)
batch = next(iter(train_loader))
print("Minibatch shapes →  x_num:", batch[0].shape, "x_cat:", batch[1].shape, "y:", batch[2].shape)

# ---------- 5. Model / hyper-parameters -------------------------------
n_classes = 2
print("Detected n_classes =", n_classes)
hyperparameters = {
    "hidden_layers_size": [128, 64],
    "activation": "relu",
    "batch_normalization": False,
    "dropout_rate": 0.1,
    "output_dim": n_classes,
    "num_numeric_features": len(num_cols),
    "learning_rate": 0.0001,
    "max_epoch": 1000,
}

embedding_sizes = {
    'region':  (num_regions, 8),
    'station': (num_stations, 16),
    'cepages': (num_cepages, 8)
}
model = TabularMLP(hyperparameters, embedding_sizes).to(device)
scope = ScopeClassifier(model, hyperparameters,steps_per_epoch=len(train_loader))

# ---------- 6. Balanced loss -------------------------------------------
cnt = Counter(y_train.cpu().numpy())
total = len(y_train)
weights = [total / cnt[cls] for cls in [0, 1]]
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
scope.criterion = criterion


# ---------- 7. Trainer workflow ----------------------------------------
trainer = TrainerClassifier(hyperparameter=hyperparameters)
trainer.set_model(model=model, device=device)
trainer.set_scope(scope=scope)
trainer.set_data(x_train=(x_num_train, x_cat_train), y_train=y_train,
                 x_valid=(x_num_valid, x_cat_valid), y_valid=y_valid)

train_acc_hist, valid_acc_hist = trainer.run()

# ---------- 8. Plot accuracy curves ------------------------------------
plt.figure()
plt.plot(train_acc_hist,  label="Train accuracy")
plt.plot(valid_acc_hist,  label="Valid accuracy")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

# ---------- 9. Quick inference on validation ---------------------------
with torch.no_grad():
    y_hat = model(x_num_valid.to(device), x_cat_valid.to(device))
    pred  = y_hat.argmax(dim=1).cpu().numpy()

print("Validation predictions shape:", pred.shape)

# ---------- 10. LightGBM baseline --------------------------------------
x_1hot_train = torch.cat([
    torch.nn.functional.one_hot(x_cat_train[:, i], num_classes=num_c).float()
    for i, num_c in enumerate([num_regions, num_stations, num_cepages])
], dim=1)

x_1hot_valid = torch.cat([
    torch.nn.functional.one_hot(x_cat_valid[:, i], num_classes=num_c).float()
    for i, num_c in enumerate([num_regions, num_stations, num_cepages])
], dim=1)


X_train = torch.cat([x_num_train, x_1hot_train], dim=1).cpu().numpy()
X_valid = torch.cat([x_num_valid, x_1hot_valid], dim=1).cpu().numpy()
y_train_np = y_train.cpu().numpy()
y_valid_np = y_valid.cpu().numpy()

lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05)
lgbm.fit(X_train, y_train_np)

lgbm_pred = lgbm.predict(X_valid)
lgbm_pred = np.asarray(lgbm_pred)
lgbm_acc = accuracy_score(y_valid_np, lgbm_pred)
print("LGBM valid acc:", lgbm_acc)

# ---------- 11. Inference on test set ----------------------------------
x_num_test, x_cat_test, y_test = test_ds.tensors
with torch.no_grad():
    y_test_hat = model(x_num_test.to(device), x_cat_test.to(device))
    test_pred  = y_test_hat.argmax(dim=1).cpu().numpy()
if test_pred.ndim > 1:
    test_pred = (test_pred > 0.5).astype(int)

if isinstance(test_pred, torch.Tensor):
    test_pred = test_pred.cpu().numpy()
if isinstance(y_test, torch.Tensor):
    y_test = y_test.cpu().numpy()

# Print evaluation metrics
test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)

print(f"\n✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test F1 Score:  {test_f1:.4f}\n")

# Optional: show full classification report
print(classification_report(y_test, test_pred))

# One-hot encoding des features catégoriques
x_1hot_test = torch.cat([
    torch.nn.functional.one_hot(x_cat_test[:, i], num_classes=num_c).float()
    for i, num_c in enumerate([num_regions, num_stations, num_cepages])
], dim=1)

# Concaténation avec les features numériques
x_lgbm_test = torch.cat([x_num_test, x_1hot_test], dim=1).numpy()

# Prédictions avec LightGBM
lgbm_test_preds = lgbm.predict(x_lgbm_test)
lgbm_test_preds = np.asarray(lgbm_test_preds)

# Labels vrais
y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

# Metrics
lgbm_test_acc = accuracy_score(y_test_np, lgbm_test_preds)
lgbm_test_f1 = f1_score(y_test_np, lgbm_test_preds)

print(f"\n🌟 LGBM Test Accuracy: {lgbm_test_acc:.4f}")
print(f"🌟 LGBM Test F1 Score:  {lgbm_test_f1:.4f}\n")
print(classification_report(y_test_np, lgbm_test_preds))

# Save the model
# torch.save(model.state_dict(), "models/mlp_wine_quality.pth")
# print("Model saved to 'models/mlp_wine_quality.pth'")
