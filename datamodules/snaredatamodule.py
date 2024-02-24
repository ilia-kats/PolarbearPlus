import os

import lightning as L
import numpy as np
import pandas as pd
import requests
import torch
from scipy.io import mmread
from torch.utils.data import DataLoader, TensorDataset, random_split


class SNAREDataModule(L.LightningDataModule):
    """Data Module for SNAREseq dataset."""

    files = [
        "adultbrainfull50_rna_outer_snareseq.mtx",
        "adultbrainfull50_atac_outer_snareseq.mtx",
        "adultbrainfull50_atac_outer_snareseq_barcodes.tsv",
        "adultbrainfull50_atac_outer_peaks.txt",
        "adultbrainfull50_rna_outer_genes.txt",
    ]
    base_url = "https://noble.gs.washington.edu/~ranz0/Polarbear/data/"

    def __init__(self, data_dir: str = "./data/snareseq"):
        super().__init__()
        self.seed = 42
        self.data_dir = data_dir
        self._num_cells = None
        self._num_peaks = None
        self._num_genes = None
        self._peaks = None
        self._genes = None

    def prepare_data(self):
        """Download data and prepare train/test/val split."""
        for filename in self.files:
            url = f"{self.base_url}{filename}"
            if os.path.isfile(os.path.join(self.data_dir, filename)):
                print(f"{filename} already downloaded")
            else:
                if os.path.isdir(self.data_dir):
                    pass
                else:
                    os.makedirs(self.data_dir)

                try:
                    r = requests.get(url, timeout=10)
                    with open(os.path.join(self.data_dir, filename), "wb") as f:
                        f.write(r.content)
                    print(f"Downloaded {filename}")
                except requests.exceptions.RequestException:
                    print(f"Failed to download {url}")

        # prepare the random train/test/val split
        cells = pd.read_csv(
            os.path.join(self.data_dir, "adultbrainfull50_atac_outer_snareseq_barcodes.tsv"), sep="\t", header=0
        )
        train, test, val = random_split(
            torch.Tensor(cells.index), [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(self.seed)
        )
        for i, name in zip([train.indices, test.indices, val.indices], ["train", "test", "val"], strict=False):
            torch.save(i, os.path.join(self.data_dir, f"{name}.pt"))

    @property
    def num_genes(self):
        """Number of genes in dataset."""
        return self._num_genes

    @property
    def num_peaks(self):
        """Number of peaks in dataset."""
        return self._num_peaks

    @property
    def num_cells(self):
        """Number of cells in dataset."""
        return self._num_cells

    @property
    def peaks(self):
        """Peak Metadata."""
        return self._peaks

    @property
    def genes(self):
        """Gene Metadata."""
        return self._genes

    # @property
    # def logbatchnorm(self):
    #    return self.logbatchnorm
    # @property
    # def logbatchvar(self):
    #    return self.logbatchvar

    def setup(self, stage: str):
        """Load Dataset and assign train/val/test datasets for use in dataloaders."""
        atac_counts = mmread(os.path.join(self.data_dir, "adultbrainfull50_atac_outer_snareseq.mtx"))
        rna_counts = mmread(os.path.join(self.data_dir, "adultbrainfull50_rna_outer_snareseq.mtx"))
        # concatenate along cell dimension, first g columns correspond to genes, last p columns correspond to peaks
        counts = np.concatenate((rna_counts.toarray(), atac_counts.toarray()), axis=1)

        self._num_cells = rna_counts.shape[0]
        self._num_peaks = atac_counts.shape[1]
        self._num_genes = rna_counts.shape[1]
        self._peaks = pd.read_csv(
            os.path.join(self.data_dir, "adultbrainfull50_atac_outer_peaks.txt"), sep="\t", header=None
        )
        self._genes = pd.read_csv(
            os.path.join(self.data_dir, "adultbrainfull50_rna_outer_genes.txt"), sep="\t", header=None
        )
        # self.logbatchnorm = None
        # self.logbatchvar = None

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_ind, val_ind = (
                torch.load(os.path.join(self.data_dir, "train.pt")),
                torch.load(os.path.join(self.data_dir, "val.pt")),
            )
            self.snare_train = TensorDataset(torch.from_numpy(counts[train_ind]))
            self.snare_val = TensorDataset(torch.from_numpy(counts[val_ind]))

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            test_ind = torch.load(os.path.join(self.data_dir, "test.pt"))
            self.snare_test = TensorDataset(torch.from_numpy(counts[test_ind]))

        if stage == "predict":
            test_ind = torch.load(os.path.join(self.data_dir, "test.pt"))
            self.snare_predict = TensorDataset(torch.from_numpy(counts[test_ind]))

    def train_dataloader(self):
        """Train Dataloader."""
        return DataLoader(self.snare_train, batch_size=32)

    def val_dataloader(self):
        """Train Dataloader."""
        return DataLoader(self.snare_val, batch_size=32)

    def test_dataloader(self):
        """Train Dataloader."""
        return DataLoader(self.snare_test, batch_size=32)

    def predict_dataloader(self):
        """Train Dataloader."""
        return DataLoader(self.snare_predict, batch_size=32)
