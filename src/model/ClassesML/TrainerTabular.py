# ClassesML/TrainerTabular.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from Utils.Utilities import Utilities            # keep your helper

class TrainerClassifier:
    def __init__(self, hyperparameter):
        self.hp = hyperparameter         # shorthand

    # setters stay the same ----------------------------------------------
    def set_model(self, model, device):
        self.model, self.device = model, device

    def set_scope(self, scope):
        self.scope = scope

    def set_data(self, x_train, y_train, x_valid, y_valid):
        # we keep the raw tensors but immediately wrap them in DataLoaders
        bs = self.hp.get("batch_size", 512)

        if isinstance(x_train, (tuple, list)):
            self.train_loader = DataLoader(
                TensorDataset(*x_train, y_train),
                batch_size=bs,
                shuffle=True,
                drop_last=False,
            )
            self.valid_loader = DataLoader(
                TensorDataset(*x_valid, y_valid),
                batch_size=bs * 2,
                shuffle=False,
            )
        else:
            self.train_loader = DataLoader(
                TensorDataset(x_train, y_train),
                batch_size=bs,
                shuffle=True,
                drop_last=False,
            )
            self.valid_loader = DataLoader(
                TensorDataset(x_valid, y_valid),
                batch_size=bs * 2,
                shuffle=False,
            )
        
    def set_data_loaders(self, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader


    # --------------------------------------------------------------------
    def _epoch_loop(self, loader, train_phase=True):
        self.model.train() if train_phase else self.model.eval()

        total_loss = total_correct = total_samples = 0
        for *features, y in loader:                    # ← générique
            features = [f.to(self.device) for f in features]
            y = y.to(self.device)

            y_hat = self.model(*features)              # MLP : 2 tensors ; RNN : 3
            loss  = self.scope.criterion(y_hat, y)

            if train_phase:
                self.scope.optimizer.zero_grad()
                loss.backward()
                self.scope.optimizer.step()

            total_loss    += loss.item() * y.size(0)
            total_correct += (y_hat.argmax(1) == y).sum().item()
            total_samples += y.size(0)

        avg_loss = total_loss / total_samples
        avg_acc  = 100.0 * total_correct / total_samples
        return avg_loss, avg_acc


    # --------------------------------------------------------------------
    def run(self):
        print("Starting training loop…")
        train_hist, valid_hist = [], []

        for epoch in range(1, self.hp["max_epoch"] + 1):
            tr_loss, tr_acc = self._epoch_loop(self.train_loader, train_phase=True)
            va_loss, va_acc = self._epoch_loop(self.valid_loader, train_phase=False)

            train_hist.append(tr_acc)
            valid_hist.append(va_acc)

            print(f"Epoch {epoch:2d}/{self.hp['max_epoch']}  "
                  f"train: loss {tr_loss:.4f} | acc {tr_acc:.2f}%   "
                  f"valid: loss {va_loss:.4f} | acc {va_acc:.2f}%")

            # scheduler / early stopping (same logic as before)
            if self.scope.scheduler:
                old_lr = self.scope.optimizer.param_groups[0]['lr']
                self.scope.scheduler.step(va_acc)
                new_lr = self.scope.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(f"LR change: {old_lr} → {new_lr} at epoch {epoch}")

                if self.scope.early_stopping:
                    if not self.scope.early_stopping.set(self.model, epoch, va_acc):
                        print("Early stopping triggered.")
                        break
        best_epoch = int(torch.tensor(valid_hist).argmax()) + 1
        print(f"\nBest validation accuracy: {max(valid_hist):.2f}% at epoch {best_epoch}")

        return train_hist, valid_hist
