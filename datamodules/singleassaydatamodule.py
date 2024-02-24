import os
import urllib.request

import lightning as L
import numpy as np
import pandas as pd
import torch
from scipy.io import mmread
from scipy.sparse import vstack
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm.auto import tqdm


class _TqdmDownload(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.update({"unit": "B", "unit_scale": True, "unit_divisor": 1024})
        super().__init__(*args, **kwargs)

    def update_to(self, nblocks=1, blocksize=1, total=-1):
        self.total = total
        self.update(nblocks * blocksize - self.n)


def _download(url, outfile, desc):
    with _TqdmDownload(desc="downloading " + desc) as t:
        urllib.request.urlretrieve(url, outfile, t.update_to)  # noqa S310


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
        idx = idx.numpy() if torch.is_tensor(idx) else idx
        data = self.counts[idx, :].toarray()
        batch = self.batch_info[idx]
        return data.squeeze(), batch


class AtacDataModule(L.LightningDataModule):
    """Data Module for single assay ATAC-seq data.

    The Data Module downloads the data from the base_url and prepares the train/val/test split.
    There is one co-assays dataset "SNARE-seq" and one single assay dataset "ATAC-seq".
    The peaks are shared between the two datasets. Each data set corresponds to one batch.
    We concatenate the two datasets along the cell dimension and we add a vector of batch indices
    to the output of our TensorDataset.

    Args:
        batch_size: Minibatch size.
        n_workers: Number of dataloader workers.
        pin_memory: Whether to use pinned memory.
        data_dir: directory to save all files
        seed: set a random seed for train/val/test split
    """

    _files = {
        "single": "adultbrainfull50_atac_outer_single.mtx",
        "snareseq": "adultbrainfull50_atac_outer_snareseq.mtx",
        "snareseq_barcodes": "adultbrainfull50_atac_outer_snareseq_barcodes.tsv",
        "single_barcodes": "adultbrainfull50_atac_outer_single_barcodes.tsv",
        "peak_annotation": "adultbrainfull50_atac_outer_peaks.txt",
    }
    _base_url = "https://noble.gs.washington.edu/~ranz0/Polarbear/data/"

    def __init__(
        self,
        batch_size: int,
        n_workers: int = 0,
        pin_memory: bool = False,
        data_dir: str = "./data/snareseq",
        seed: int = 42,
    ):
        super().__init__()
        self._batch_size = batch_size
        self._n_workers = n_workers
        self._pin_memory = pin_memory

        self._seed = seed
        self._data_dir = data_dir
        self._num_cells = None
        self._num_genes = None
        self._chr_idx = None

        self._dset_train = self._dset_test = self._dset_val = None

    def prepare_data(self):
        """Download data and prepare train/test/val split."""
        for filedesc, filename in self._files.items():
            url = f"{self._base_url}{filename}"
            filepath = os.path.join(self._data_dir, filename)
            if os.path.isfile(filepath):
                print(f"{filepath} already downloaded")
            else:
                os.makedirs(self._data_dir, exist_ok=True)
                _download(url, filepath, filedesc)

    @property
    def num_peaks(self):
        """Number of peaks in the dataset."""
        return self._num_peaks

    @property
    def num_cells(self):
        """Number of cells in the dataset."""
        return self._num_cells

    @property
    def num_batches(self):
        """Number of experimental batches (datasets)."""
        return 2

    @property
    def chromosome_indices(self):
        """List of index ranges in the data matrix belonging to separate chromosomes."""
        return self._chr_idx

    def setup(self, stage: str | None = None):
        """Load Dataset and assign train/val/test datasets for use in dataloaders."""
        if self._chr_idx is None or self._dset_train is None or self._dset_val is None or self._dset_test is None:
            atac_counts1 = mmread(os.path.join(self._data_dir, self._files["snareseq"]))
            atac_counts2 = mmread(os.path.join(self._data_dir, self._files["single"]))
            # concatenate along cell dimension, first g columns correspond to genes, last p columns correspond to peaks
            counts = vstack((atac_counts1, atac_counts2))
            counts = counts.astype(np.float32)
            counts = counts.tocsr()  # convert to compressed sparse row format
            counts.data[:] = 1

            # create batch index for the two data sets
            batch_info = torch.cat(
                (
                    torch.zeros(atac_counts1.shape[0], dtype=torch.int64),
                    torch.ones(atac_counts2.shape[0], dtype=torch.int64),
                )
            )

            dset = AtacCounts(counts, batch_info)
            self._dset_train, self._dset_val, self._dset_test = random_split(
                dset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(self._seed)
            )

            self._num_cells = counts.shape[0]
            self._num_peaks = counts.shape[1]
            peaks = pd.read_table(
                os.path.join(self._data_dir, self._files["peak_annotation"]),
                sep=":",
                header=None,
                names=["chr", "range"],
            )
            chromosomes, peak_indices, peak_counts = np.unique(
                peaks.chr.to_numpy(), return_index=True, return_counts=True
            )
            chrorder = np.argsort(peak_indices)
            chromosomes, peak_indices, peak_counts = (
                chromosomes[chrorder],
                peak_indices[chrorder],
                peak_counts[chrorder],
            )
            self._chr_idx = [(idx, idx + cnt) for idx, cnt in zip(peak_indices, peak_counts, strict=False)]

    def _get_dataloader(self, dset):
        return DataLoader(dset, batch_size=self._batch_size, num_workers=self._n_workers, pin_memory=self._pin_memory)

    def train_dataloader(self):
        """Train dataloader."""
        return self._get_dataloader(self._dset_train)

    def val_dataloader(self):
        """Validation dataloader."""
        return self._get_dataloader(self._dset_val)

    def test_dataloader(self):
        """Test dataloader."""
        return self._get_dataloader(self._dset_test)

    def predict_dataloader(self):
        """Predict dataloader."""
        return self._get_dataloader(self._dset_test)


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
            filepath = os.path.join(self.data_dir, filename)
            if os.path.isfile(filepath):
                print(f"{filepath} already downloaded")
            else:
                os.makedirs(self.data_dir, exist_ok=True)
                _download(url, filepath, filepath)

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
            torch.Tensor(cells.index), [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(self._seed)
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
