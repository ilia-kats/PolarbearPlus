import logging
import os
from typing import Literal

import numpy as np
import torch
from scipy.io import mmread
from torch.utils.data import StackDataset, random_split

from .utils import PolarbearDataModuleBase, SparseDataset

_logger = logging.getLogger(__name__)


class SNAREDataModule(PolarbearDataModuleBase):
    """Data Module for SNAREseq dataset."""

    _files = {
        "rna": "adultbrainfull50_rna_outer_snareseq.mtx",
        "atac": "adultbrainfull50_atac_outer_snareseq.mtx",
        "barcodes": "adultbrainfull50_atac_outer_snareseq_barcodes.tsv",
        "peaks": "adultbrainfull50_atac_outer_peaks.txt",
        "genes": "adultbrainfull50_rna_outer_genes.txt",
    }

    def __init__(
        self,
        direction: Literal["rna2atac", "atac2rna"],
        batch_size: int,
        n_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        data_dir: str = "./data/snareseq",
    ):
        super().__init__(batch_size, n_workers, pin_memory, persistent_workers, data_dir)
        self._direction = direction
        self._num_cells = None
        self._num_genes = None
        self._num_peaks = None

    @property
    def num_genes(self):
        """Number of genes in the dataset."""
        if self._num_genes is None:
            self._init()
        return self._num_genes

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
        return 1

    def setup(self, stage: str):
        """Load Dataset and assign train/val/test datasets for use in dataloaders."""
        if (
            self._num_cells is None
            or self._num_genes is None
            or self._num_peaks is None
            or self._dset_train is None
            or self._dset_test is None
            or self._dset_val is None
        ):
            _logger.info("reading data...")
            atac_counts = mmread(os.path.join(self._data_dir, self._files["atac"])).tocsr().astype(np.float32)
            rna_counts = mmread(os.path.join(self._data_dir, self._files["rna"])).tocsr().astype(np.float32)

            atac_counts.data[:] = 1

            atac_dset = SparseDataset(atac_counts)
            rna_dset = SparseDataset(rna_counts)
            batch_idx = torch.zeros((atac_counts.shape[0],), dtype=torch.int64)

            dset = (
                StackDataset(rna_dset, batch_idx, atac_dset, batch_idx)
                if self._direction == "rna2atac"
                else StackDataset(atac_dset, batch_idx, rna_dset, batch_idx)
            )

            self._dset_train, self._dset_val, self._dset_test = random_split(dset, [0.6, 0.2, 0.2])
