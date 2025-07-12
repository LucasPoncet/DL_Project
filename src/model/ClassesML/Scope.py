import torch.nn as nn 
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ClassesML.EarlyStopper import EarlyStopper
from torch.optim.lr_scheduler import OneCycleLR



class ScopeClassifier:
    def __init__(self, model, hyperparameters,steps_per_epoch = None):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=1e-5)
        if "patience_lr" in hyperparameters:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max',
                                               patience=hyperparameters["patience_lr"],
                                               factor=0.1)
        elif "cycle_lr" in hyperparameters:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=hyperparameters["cycle_lr"],
                steps_per_epoch=steps_per_epoch,
                epochs=hyperparameters["max_epoch"]
            )
        else:
            self.scheduler = None
        
        if "early_stopping" in hyperparameters:
            self.early_stopping = EarlyStopper(hyperparameters=hyperparameters)
        else:
            self.early_stopping = None
        

class ScopeAutoEncoder:
    def __init__(self, model, hyperparameters):
        self.criterion = nn.MSELoss()
        encoder_parameters = list(model.encoder.parameters())
        decoder_parameters = list(model.decoder.parameters())
        autoencoder_parameters = encoder_parameters + decoder_parameters

        self.optimizer = optim.Adam(autoencoder_parameters, lr=hyperparameters["learning_rate"])
        
class ScopeGAN:
    def __init__(self, generator, discriminator, hyper_models):
        self.criterion_generator = nn.MSELoss()
        self.criterion_discriminator = nn.BCELoss()
        self.optimizer_generator = optim.Adam(generator.parameters(), lr=hyper_models["learning_rate"])
        self.optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=hyper_models["learning_rate"])
        