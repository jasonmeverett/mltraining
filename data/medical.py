import kagglehub
from data.base import DataModule
import pandas as pd
import os 
import numpy as np 
import torch

from torch.utils.data import TensorDataset, DataLoader


class HeartFailurePredictionDataset(DataModule):
    """
    https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
    """

    def __init__(self, train_test_split=0.1, batch_size=32, seed=42):
        super().__init__()
        self.batch_size = batch_size
        self.path = kagglehub.dataset_download(
            "fedesoriano/heart-failure-prediction"
        )
        self.df = pd.read_csv(os.path.join(self.path, 'heart.csv'))
        self.input_map = {}
        self.cols = list(self.df.columns)
        self.categorical_cols = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]
        self.numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
        for cType in self.categorical_cols:
            cat = pd.Categorical(self.df[cType])
            self.input_map[cType] = {cat.categories[i]: i for i in range(len(cat.categories))}
            self.df[cType] = cat.codes

        targetCol = "HeartDisease"
        self.X = self.df.drop(columns=[targetCol])
        self.y = self.df[targetCol]

        # Compute stats
        self.stats = {cName: {"mean": self.df[cName].mean().item(), "std": self.df[cName].std().item()} for cName in self.numeric_cols}
        
        # Split training and validation
        N = len(self.df)
        indices = np.arange(N)
        split_idx = int(N*(1.0 - train_test_split))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        def split(arr):
            return arr[train_idx], arr[val_idx]

        self.splits = {}
        for col in self.cols:
            cTrain, cSplit = split(self.df[col])
            self.splits[col] = (torch.from_numpy(cTrain.values.astype("float32")), torch.from_numpy(cSplit.values.astype("float32")))

        self.train_dataset = TensorDataset(*[self.splits[x][0] for x in self.splits])
        self.val_dataset = TensorDataset(*[self.splits[x][1] for x in self.splits])


    def prepare_batch(self, *X):
        return X

    def train_dataloader(self):
        """
        Return a tensorloader for training data (should shuffle)
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        """
        Return a tensorloader for validation data (should not shuffle)
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )




