import logging
import os

import numpy as np
import pandas as pd
import torch
from scipy.io import mmread
from torch.utils.data import ConcatDataset, StackDataset, Subset

from .utils import PolarbearDataModuleBase, SparseDataset

_logger = logging.getLogger(__name__)


class AtacDataModule(PolarbearDataModuleBase):
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
    def num_batches(self):
        """Number of experimental batches (datasets)."""
        return 3

    @property
    def chromosome_indices(self):
        """List of index ranges in the data matrix belonging to separate chromosomes."""
        if self._chr_idx is None:
            self._init()
        return self._chr_idx

    def setup(self, stage: str | None = None):
        """Load Dataset and assign train/val/test datasets for use in dataloaders."""
        if self._chr_idx is None:
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

        if stage is not None:
            _logger.info("reading data...")
            atac_counts1 = mmread(os.path.join(self._data_dir, self._files["atac_snareseq"])).tocsr().astype(np.float32)
            atac_counts2 = mmread(os.path.join(self._data_dir, self._files["atac_single"])).tocsr().astype(np.float32)
            atac_counts1.data[:], atac_counts2.data[:] = 1, 1

            atac_counts2_datasets = (
                mmread(os.path.join(self._data_dir, self._files["atac_single_dataset"])).tocsr().indices
            )
            atac_counts2_datasets -= atac_counts2_datasets.min()

            split = {k: np.loadtxt(os.path.join(self._data_dir, v), dtype=int) for k, v in self._split_files.items()}

            snare_dset = StackDataset(
                SparseDataset(atac_counts1), torch.zeros((atac_counts1.shape[0],), dtype=torch.int64)
            )
            snare_train, self._dset_val, self._dset_test = (
                Subset(snare_dset, split["train"]),
                Subset(snare_dset, split["val"]),
                Subset(snare_dset, split["test"]),
            )

            # add the single assay dataset to the training set
            single_dset = StackDataset(
                SparseDataset(atac_counts2), torch.as_tensor(atac_counts2_datasets + 1, dtype=torch.int64)
            )
            self._dset_train = ConcatDataset([snare_train, single_dset])

            self._num_peaks = atac_counts1.shape[1]


class RnaDataModule(PolarbearDataModuleBase):
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
            rna_counts1 = mmread(os.path.join(self._data_dir, self._files["rna_snareseq"])).tocsr().astype(np.float32)
            rna_counts2 = mmread(os.path.join(self._data_dir, self._files["rna_single"])).tocsr().astype(np.float32)

            split = {k: np.loadtxt(os.path.join(self._data_dir, v), dtype=int) for k, v in self._split_files.items()}

            snare_dset = StackDataset(
                SparseDataset(rna_counts1), torch.zeros((rna_counts1.shape[0],), dtype=torch.int64)
            )
            snare_train, self._dset_val, self._dset_test = (
                Subset(snare_dset, split["train"]),
                Subset(snare_dset, split["val"]),
                Subset(snare_dset, split["test"]),
            )

            # add the single assay dataset to the training set
            single_dset = StackDataset(
                SparseDataset(rna_counts2), torch.ones((rna_counts2.shape[0],), dtype=torch.int64)
            )
            self._dset_train = ConcatDataset([snare_train, single_dset])

            self._num_genes = rna_counts1.shape[1]
            self._genes = pd.read_csv(os.path.join(self._data_dir, self._files["genenames"]), sep="\t", header=None)

            # Compute mean and variance of library size for each batch
            library_size1, library_size2 = (
                np.log(rna_counts1[split["train"], :].sum(axis=1).A1),
                np.log(rna_counts2.sum(axis=1).A1),
            )
            self._logbatchmean = np.array([library_size1.mean(), library_size2.mean()])
            self._logbatchvar = np.array([library_size1.var(), library_size2.var()])
