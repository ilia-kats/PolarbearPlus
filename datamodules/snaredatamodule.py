import os

import lightning as L
import numpy as np
import pandas as pd
import requests
import torch
from scipy.io import mmread
from torch.utils.data import DataLoader, TensorDataset, random_split


class RnaDataModule(L.LightningDataModule):
    """Data Module for single assay RNA-seq data.

    The Data Module downloads the data from the base_url and prepares the train/val/test split.
    There is one co-assays dataset "SNARE-seq" and one single assay dataset "RNA-seq".
    The genes are shared between the two datasets. Each data set corresponds to one batch.
    We concatenate the two datasets along the cell dimension and we add a vector of batch indices
    to the output of our TensorDataset.

    Attributes:
    ----------
    files : list
        a list of strings containing the names of the files to download
    base_url : str
        the base url to download the files from
    seed : int
        set a random seed for train/val/test split
    data_dir : str
        directory to save all files
    num_cells : int
        number of cells in the dataset
    num_genes : int
        number of genes in the dataset
    genes : pd.DataFrame
        gene metadata (gene names)
    logbatchmean : np.array
        mean of library size for each batch
    logbatchvar : np.array
        variance of library size for each batch

    """

    files = [
        "adultbrainfull50_rna_outer_snareseq.mtx",  # SNARE-seq
        "adultbrainfull50_rna_outer_single.mtx",  # Single assay RNA-seq
        # "adultbrainfull50_atac_outer_snareseq.mtx",
        "adultbrainfull50_atac_outer_snareseq_barcodes.tsv",  # SNARE-seq cells
        "adultbrainfull50_rna_outer_single_barcodes.tsv",  # single assay cells
        # "adultbrainfull50_atac_outer_peaks.txt",
        "adultbrainfull50_rna_outer_genes.txt",
    ]

    base_url = "https://noble.gs.washington.edu/~ranz0/Polarbear/data/"

    def __init__(self, data_dir: str = "./data/snareseq"):
        """Parameters.

        ----------

        seed : int
            set a random seed for train/val/test split
        data_dir : str
            directory to save all files
        num_cells : int
            number of cells in the dataset
        num_genes : int
            number of genes in the dataset
        genes : pd.DataFrame
            gene metadata (gene names)
        logbatchmean : np.array
            mean of library size for each batch
        logbatchvar : np.array
            variance of library size for each batch
        """
        super().__init__()
        self.seed = 42
        self.data_dir = data_dir
        self._num_cells = None
        self._num_genes = None
        self._genes = None
        self._logbatchmean = None
        self._logbatchvar = None

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
        cells1 = pd.read_csv(
            os.path.join(self.data_dir, "adultbrainfull50_atac_outer_snareseq_barcodes.tsv"), sep="\t", header=0
        )
        cells2 = pd.read_csv(
            os.path.join(self.data_dir, "adultbrainfull50_rna_outer_single_barcodes.tsv"), sep="\t", header=0
        )
        cells1["batch"]
        cells = pd.concat([cells1, cells2], axis=0)

        train, test, val = random_split(
            torch.Tensor(cells.index), [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(self.seed)
        )
        for i, name in zip(
            [train.indices, test.indices, val.indices], ["rna_train", "rna_test", "rna_val"], strict=False
        ):
            torch.save(i, os.path.join(self.data_dir, f"{name}.pt"))

    @property
    def num_genes(self):
        """Number of genes in dataset."""
        return self._num_genes

    @property
    def num_cells(
        self,
    ):
        """Number of cells in dataset."""
        return self._num_cells

    @property
    def genes(self):
        """Gene Metadata."""
        return self._genes

    @property
    def logbatchnorm(self):
        """Mean of library size for each batch."""
        return self._logbatchnorm

    @property
    def logbatchvar(self):
        """Variance of library size for each batch."""
        return self._logbatchvar

    def setup(self, stage: str):
        """Load Dataset and assign train/val/test datasets for use in dataloaders."""
        rna_counts1 = mmread(os.path.join(self.data_dir, "adultbrainfull50_rna_outer_snareseq.mtx"))
        rna_counts2 = mmread(os.path.join(self.data_dir, "adultbrainfull50_rna_outer_single.mtx"))
        # concatenate along cell dimension, first g columns correspond to genes, last p columns correspond to peaks
        counts = torch.from_numpy(
            np.concatenate((rna_counts1.toarray(), rna_counts2.toarray()), axis=0)
        )  # cells x genes
        batch_info = torch.from_numpy(
            np.concatenate((np.zeros(rna_counts1.shape[0]), np.ones(rna_counts2.shape[0])), axis=0)
        )
        print(batch_info.shape, counts.shape)
        self._num_cells = counts.shape[0]
        self._num_genes = counts.shape[1]
        self._genes = pd.read_csv(
            os.path.join(self.data_dir, "adultbrainfull50_rna_outer_genes.txt"), sep="\t", header=None
        )

        # Compute mean and variance of library size for each batch
        library_size1, library_size2 = np.log(rna_counts1.sum(axis=1)), np.log(rna_counts2.sum(axis=1))
        print(library_size1.shape, library_size2.shape)
        self._logbatchmean = np.array([library_size1.mean(), library_size2.mean()])
        self._logbatchvar = np.array([library_size1.var(), library_size2.var()])
        print(self._logbatchmean, self._logbatchvar)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_ind, val_ind = (
                torch.load(os.path.join(self.data_dir, "rna_train.pt")),
                torch.load(os.path.join(self.data_dir, "rna_val.pt")),
            )
            self.snare_train = TensorDataset(counts[train_ind], batch_info[train_ind])
            self.snare_val = TensorDataset(counts[val_ind], batch_info[val_ind])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            test_ind = torch.load(os.path.join(self.data_dir, "rna_test.pt"))
            self.snare_test = TensorDataset(counts[test_ind], batch_info[test_ind])

        if stage == "predict":
            test_ind = torch.load(os.path.join(self.data_dir, "rna_test.pt"))
            self.snare_predict = TensorDataset(counts[test_ind], batch_info[test_ind])

    def train_dataloader(self):
        """Train Dataloader."""
        return DataLoader(self.snare_train, batch_size=32)

    def val_dataloader(self):
        """Validation Dataloader."""
        return DataLoader(self.snare_val, batch_size=32)

    def test_dataloader(self):
        """Test Dataloader."""
        return DataLoader(self.snare_test, batch_size=32)

    def predict_dataloader(self):
        """Predict Dataloader."""
        return DataLoader(self.snare_predict, batch_size=32)


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
