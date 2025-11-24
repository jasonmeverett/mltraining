from abc import ABC, abstractmethod


class DataModule(ABC):

    def __init__(self):
        return

    @abstractmethod
    def train_dataloader(self):
        pass 

    @abstractmethod
    def val_dataloader(self):
        pass 

    def prepare_batch(self, X):
        return X