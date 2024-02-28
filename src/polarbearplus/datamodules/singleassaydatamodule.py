import logging
import os
import urllib.request

import lightning as L
import numpy as np
import pandas as pd
import torch
from scipy.io import mmread
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from tqdm.auto import tqdm

_logger = logging.getLogger(__name__)


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


class SparseDataset(Dataset):
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


class _DataModuleBase(L.LightningDataModule):
    _base_url = "https://noble.gs.washington.edu/~ranz0/Polarbear/data/"
    _files = {}

    def __init__(
        self,
        batch_size: int,
        n_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        data_dir: str = "./data/snareseq",
    ):
        super().__init__()

        self._batch_size = batch_size
        self._n_workers = n_workers
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers

        self._data_dir = data_dir
        self._dset_train = self._dset_test = self._dset_val = None

    def _init(self):
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        """Download data and prepare train/test/val split."""
        for filedesc, filename in self._files.items():
            url = f"{self._base_url}{filename}"
            filepath = os.path.join(self._data_dir, filename)
            if not os.path.isfile(filepath):
                os.makedirs(self._data_dir, exist_ok=True)
                _download(url, filepath, filedesc)

    def _get_dataloader(self, dset, shuffle=False):
        return DataLoader(
            dset,
            shuffle=shuffle,
            batch_size=self._batch_size,
            num_workers=self._n_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
        )

    def train_dataloader(self):
        """Train dataloader."""
        return self._get_dataloader(self._dset_train, True)

    def val_dataloader(self):
        """Validation dataloader."""
        return self._get_dataloader(self._dset_val)

    def test_dataloader(self):
        """Test dataloader."""
        return self._get_dataloader(self._dset_test)

    def predict_dataloader(self):
        """Predict dataloader."""
        return self._get_dataloader(self._dset_test)


class AtacDataModule(_DataModuleBase):
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
        persistent_workers: Whether to not shut down the worker processes after every epoch.
        data_dir: directory to save all files
    """

    _files = {
        "single": "adultbrainfull50_atac_outer_single.mtx",
        "snareseq": "adultbrainfull50_atac_outer_snareseq.mtx",
        "snareseq_barcodes": "adultbrainfull50_atac_outer_snareseq_barcodes.tsv",
        "single_barcodes": "adultbrainfull50_atac_outer_single_barcodes.tsv",
        "peak_annotation": "adultbrainfull50_atac_outer_peaks.txt",
    }

    def __init__(
        self,
        batch_size: int,
        n_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        data_dir: str = "./data/snareseq",
    ):
        super().__init__(batch_size, n_workers, pin_memory, persistent_workers, data_dir)
        self._num_cells = None
        self._num_genes = None
        self._chr_idx = None

    @property
    def num_peaks(self):
        """Number of peaks in the dataset."""
        if self._num_peaks is None:
            self._init()
        return self._num_peaks

    @property
    def num_cells(self):
        """Number of cells in the dataset."""
        if self._num_cells is None:
            self._init()
        return self._num_cells

    @property
    def num_batches(self):
        """Number of experimental batches (datasets)."""
        return 2

    @property
    def chromosome_indices(self):
        """List of index ranges in the data matrix belonging to separate chromosomes."""
        if self._chr_idx is None:
            self._init()
        return self._chr_idx

    def setup(self, stage: str | None = None):
        """Load Dataset and assign train/val/test datasets for use in dataloaders."""
        if self._chr_idx is None or self._dset_train is None or self._dset_val is None or self._dset_test is None:
            _logger.info("reading data...")
            atac_counts1 = mmread(os.path.join(self._data_dir, self._files["snareseq"])).tocsr().astype(np.float32)
            atac_counts2 = mmread(os.path.join(self._data_dir, self._files["single"])).tocsr().astype(np.float32)
            atac_counts1.data[:], atac_counts2.data[:] = 1, 1

            # create a random split of the co-assay data
            snare_dset = SparseDataset(atac_counts1, torch.zeros(atac_counts1.shape[0], dtype=torch.int64))
            snare_train, self._dset_val, self._dset_test = random_split(snare_dset, [0.6, 0.2, 0.2])

            # add the single assay dataset to the training set
            single_dset = SparseDataset(atac_counts2, torch.ones(atac_counts2.shape[0], dtype=torch.int64))
            self._dset_train = ConcatDataset([snare_train, single_dset])

            self._num_cells = len(snare_dset) + len(single_dset)
            self._num_peaks = atac_counts1.shape[1]
            peaks = pd.read_table(
                os.path.join(self._data_dir, self._files["peak_annotation"]),
                sep=":",
                header=None,
                names=["chr", "range"],
            )
            chromosomes, peak_indices, peak_counts = np.unique(peaks.chr.to_numpy(), a=True, return_counts=True)
            chrorder = np.argsort(peak_indices)
            chromosomes, peak_indices, peak_counts = (
                chromosomes[chrorder],
                peak_indices[chrorder],
                peak_counts[chrorder],
            )
            self._chr_idx = [(idx, idx + cnt) for idx, cnt in zip(peak_indices, peak_counts, strict=False)]


class RnaDataModule(_DataModuleBase):
    """Data Module for single assay RNA-seq data.

    The Data Module downloads the data from the base_url and prepares the train/val/test split.
    There is one co-assays dataset "SNARE-seq" and one single assay dataset "RNA-seq".
    The genes are shared between the two datasets. Each data set corresponds to one batch.
    We concatenate the two datasets along the cell dimension and we add a vector of batch indices
    to the output of our TensorDataset.

    Args:
        batch_size: Minibatch size.
        n_workers: Number of dataloader workers.
        pin_memory: Whether to use pinned memory.
        persistent_workers: Whether to not shut down the worker processes after every epoch.
        data_dir: directory to save all files
    """

    _files = {
        "snareseq": "adultbrainfull50_rna_outer_snareseq.mtx",  # SNARE-seq
        "single": "adultbrainfull50_rna_outer_single.mtx",  # Single assay RNA-seq
        "snareseq_barcodes": "adultbrainfull50_rna_outer_snareseq_barcodes.tsv",  # SNARE-seq cells
        "single_barcodes": "adultbrainfull50_rna_outer_single_barcodes.tsv",  # single assay cells
        "genenames": "adultbrainfull50_rna_outer_genes.txt",
    }

    def __init__(
        self,
        batch_size: int,
        n_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        data_dir: str = "./data/snareseq",
    ):
        super().__init__(batch_size, n_workers, pin_memory, persistent_workers, data_dir)
        self._num_cells = None
        self._num_genes = None
        self._genes = None
        self._logbatchmean = None
        self._logbatchvar = None

    @property
    def num_genes(self):
        """Number of genes in dataset."""
        if self._num_genes is None:
            self._init()
        return self._num_genes

    @property
    def num_cells(self):
        """Number of cells in dataset."""
        if self._num_cells is None:
            self._init()
        return self._num_cells

    @property
    def num_batches(self):
        """Number of experimental batches (datasets)."""
        return 2

    @property
    def genes(self):
        """Gene Metadata."""
        if self._genes is None:
            self._init()
        return self._genes

    @property
    def logbatchmean(self):
        """Mean of library size for each batch."""
        if self._logbatchmean is None:
            self._init()
        return self._logbatchmean

    @property
    def logbatchvar(self):
        """Variance of library size for each batch."""
        if self._logbatchvar is None:
            self._init()
        return self._logbatchvar

    def setup(self, stage: str | None = None):
        """Load Dataset and assign train/val/test datasets for use in dataloaders."""
        if self._genes is None or self._dset_train is None or self._dset_val is None or self._dset_test is None:
            _logger.info("reading data...")
            rna_counts1 = mmread(os.path.join(self._data_dir, self._files["snareseq"])).tocsr().astype(np.float32)
            rna_counts2 = mmread(os.path.join(self._data_dir, self._files["single"])).tocsr().astype(np.float32)

            # create a random split of the co-assay data
            snare_dset = SparseDataset(rna_counts1, torch.zeros(rna_counts1.shape[0], dtype=torch.int64))
            snare_train, self._dset_val, self._dset_test = random_split(snare_dset, [0.6, 0.2, 0.2])

            # add the single assay dataset to the training set
            single_dset = SparseDataset(rna_counts2, torch.ones(rna_counts2.shape[0], dtype=torch.int64))
            self._dset_train = ConcatDataset([snare_train, single_dset])

            self._num_cells = len(snare_dset) + len(single_dset)
            self._num_genes = rna_counts1.shape[1]
            self._genes = pd.read_csv(os.path.join(self._data_dir, self._files["genenames"]), sep="\t", header=None)

            # Compute mean and variance of library size for each batch
            library_size1, library_size2 = np.log(rna_counts1.sum(axis=1).A1), np.log(rna_counts2.sum(axis=1).A1)
            self._logbatchmean = np.array([library_size1.mean(), library_size2.mean()])
            self._logbatchvar = np.array([library_size1.var(), library_size2.var()])
