import os

import lightning as L
import numpy as np
import pandas as pd
import requests
import torch
from scipy.io import mmread
from scipy.sparse import vstack
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


class AtacCounts(Dataset):
    """ATAC-seq sparse count matrix dataloader.

    The counts are stored in a sparse matrix and the batch info is stored in a tensor.
    When iterating throught the dataloader, the data is returned as a dense tensor.
    """

    def __init__(self, counts, batch_info):
        """Arguments.

        counts: csr matrix
            sparse matrix of ATAC-seq counts
        batch_info: torch.Tensor
            vector of batch info for each cells
        """
        self.counts = counts
        self.batch_info = batch_info

    def __len__(self):
        """Return the number of cells in the dataset."""
        return self.counts.shape[0]

    def __getitem__(self, idx):
        """Return the data and batch info for idx number of samples."""
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        data = torch.Tensor(self.counts[idx, :].toarray())
        batch = self.batch_info[idx]
        return data.squeeze(), batch


class AtacDataModule(L.LightningDataModule):
    """Data Module for single assay ATAC-seq data.

    The Data Module downloads the data from the base_url and prepares the train/val/test split.
    There is one co-assays dataset "SNARE-seq" and one single assay dataset "ATAC-seq".
    The peaks are shared between the two datasets. Each data set corresponds to one batch.
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
    num_peaks : int
        number of peak in the dataset
    peaks : pd.DataFrame
        peak metadata (genomic coordiantes)
    logbatchmean : np.array
        mean of library size for each batch
    logbatchvar : np.array
        variance of library size for each batch

    """

    files = [
        "adultbrainfull50_atac_outer_single.mtx",  # Single assay RNA-seq
        "adultbrainfull50_atac_outer_snareseq.mtx",
        "adultbrainfull50_atac_outer_snareseq_barcodes.tsv",  # SNARE-seq cells
        "adultbrainfull50_atac_outer_single_barcodes.tsv",  # single assay cells
        "adultbrainfull50_atac_outer_peaks.txt",
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
            os.path.join(self.data_dir, "adultbrainfull50_atac_outer_single_barcodes.tsv"), sep="\t", header=0
        )
        cells1["batch"]
        cells = pd.concat([cells1, cells2], axis=0)

        train, test, val = random_split(
            torch.Tensor(cells.index), [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(self.seed)
        )
        for i, name in zip(
            [train.indices, test.indices, val.indices], ["atac_train", "atac_test", "atac_val"], strict=False
        ):
            torch.save(i, os.path.join(self.data_dir, f"{name}.pt"))

    @property
    def num_peaks(self):
        """Number of peaks in the dataset."""
        return self._num_peaks

    @property
    def num_cells(
        self,
    ):
        """Number of cells in the dataset."""
        return self._num_cells

    @property
    def peaks(self):
        """Peak metadata (genomic coordinates)."""
        return self._peaks

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
        atac_counts1 = mmread(os.path.join(self.data_dir, "adultbrainfull50_atac_outer_snareseq.mtx"))
        atac_counts2 = mmread(os.path.join(self.data_dir, "adultbrainfull50_atac_outer_single.mtx"))
        # concatenate along cell dimension, first g columns correspond to genes, last p columns correspond to peaks
        counts = vstack((atac_counts1, atac_counts2))
        counts = counts.astype(np.float32)
        counts = counts.tocsr()  # convert to compressed sparse row format

        # create batch index for the two data sets
        batch_info = torch.from_numpy(
            np.concatenate((np.zeros(atac_counts1.shape[0]), np.ones(atac_counts2.shape[0])), axis=0)
        )

        self._num_cells = counts.shape[0]
        self._num_peaks = counts.shape[1]
        self._peaks = pd.read_csv(
            os.path.join(self.data_dir, "adultbrainfull50_atac_outer_peaks.txt"), sep="\t", header=None
        )

        # Compute mean and variance of library size for each batch
        library_size1, library_size2 = np.log(atac_counts1.sum(axis=1)), np.log(atac_counts2.sum(axis=1))
        self._logbatchmean = np.array([library_size1.mean(), library_size2.mean()])
        self._logbatchvar = np.array([library_size1.var(), library_size2.var()])

        if stage == "fit":
            # Assign train/val datasets for use in dataloaders
            train_ind, val_ind = (
                torch.load(os.path.join(self.data_dir, "atac_train.pt")),
                torch.load(os.path.join(self.data_dir, "atac_val.pt")),
            )
            print(counts[train_ind].shape, batch_info[train_ind].shape)
            print(train_ind.shape, val_ind.shape)
            self.atac_train = AtacCounts(counts[train_ind], batch_info[train_ind])
            self.atac_val = AtacCounts(counts[val_ind], batch_info[val_ind])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            test_ind = torch.load(os.path.join(self.data_dir, "atac_test.pt"))
            self.atac_test = AtacCounts(counts[test_ind], batch_info[test_ind])

        if stage == "predict":
            test_ind = torch.load(os.path.join(self.data_dir, "atac_test.pt"))
            self.atac_predict = AtacCounts(counts[test_ind], batch_info[test_ind])

    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(self.atac_train, batch_size=32)

    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(self.atac_val, batch_size=32)

    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(self.atac_test, batch_size=32)

    def predict_dataloader(self):
        """Predict dataloader."""
        return DataLoader(self.atac_predict, batch_size=32)


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
            self.rna_train = TensorDataset(counts[train_ind], batch_info[train_ind])
            self.rna_val = TensorDataset(counts[val_ind], batch_info[val_ind])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            test_ind = torch.load(os.path.join(self.data_dir, "rna_test.pt"))
            self.rna_test = TensorDataset(counts[test_ind], batch_info[test_ind])

        if stage == "predict":
            test_ind = torch.load(os.path.join(self.data_dir, "rna_test.pt"))
            self.rna_predict = TensorDataset(counts[test_ind], batch_info[test_ind])

    def train_dataloader(self):
        """Train Dataloader."""
        return DataLoader(self.rna_train, batch_size=32)

    def val_dataloader(self):
        """Validation Dataloader."""
        return DataLoader(self.rna_val, batch_size=32)

    def test_dataloader(self):
        """Test Dataloader."""
        return DataLoader(self.rna_test, batch_size=32)

    def predict_dataloader(self):
        """Predict Dataloader."""
        return DataLoader(self.rna_predict, batch_size=32)
