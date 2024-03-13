import os
import urllib.request

import lightning as L
import numpy as np
import pandas as pd
import torch
from scipy.sparse import spmatrix
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class _TqdmDownload(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.update({"unit": "B", "unit_scale": True, "unit_divisor": 1024})
        super().__init__(*args, **kwargs)

    def update_to(self, nblocks=1, blocksize=1, total=-1):
        self.total = total
        self.update(nblocks * blocksize - self.n)


def download(url: str, outfile: str, desc: str):
    """Download a file with a nice progress bar.

    Args:
        url: URL of the file.
        outfile: Local path to save the file to.
        desc: Description to add to the progress bar.
    """
    with _TqdmDownload(desc="downloading " + desc) as t:
        urllib.request.urlretrieve(url, outfile, t.update_to)  # noqa S310


class SparseDataset(Dataset):
    """Sparse matrix dataloader.

    The data are stored in a sparse matrix. Each item is returned as a dense tensor.

    Args:
        counts: Sparse data matrix.
    """

    def __init__(self, counts: spmatrix):
        self._counts = counts

    def __len__(self):
        """Return the number of cells in the dataset."""
        return self._counts.shape[0]

    def __getitem__(self, idx):
        """Return the data and batch info for idx number of samples."""
        idx = idx.numpy() if torch.is_tensor(idx) else idx
        return self._counts[idx, :].toarray().squeeze()


class PolarbearDataModuleBase(L.LightningDataModule):
    """Base class for datamodules working with data from the Polarbear paper.

    Args:
        batch_size: Size of each minibatch.
        n_worers: Number of worker processes.
        pin_memory: Whether to use pinned memory.
        persistent_workers: Whether to keep the worker processes alive across epochs.
        data_dir: Path to local directory where the data are stored.
    """

    _base_url = "https://noble.gs.washington.edu/~ranz0/Polarbear/data/"
    _files = {
        "atac_single": "adultbrainfull50_atac_outer_single.mtx",  # Single assay ATACseq
        "atac_snareseq": "adultbrainfull50_atac_outer_snareseq.mtx",  # SNARE-seq ATACseq
        "rna_snareseq": "adultbrainfull50_rna_outer_snareseq.mtx",  # SNARE-seq RNAseq
        "rna_single": "adultbrainfull50_rna_outer_single.mtx",  # Single assay RNAseq
        "snareseq_barcodes": "adultbrainfull50_rna_outer_snareseq_barcodes.tsv",  # SNARE-seq cells
        "single_barcodes": "adultbrainfull50_rna_outer_single_barcodes.tsv",  # single assay cells
        "peak_annotation": "adultbrainfull50_atac_outer_peaks.txt",  # ATACseq peaks
        "genenames": "adultbrainfull50_rna_outer_genes.txt",  # RNAseq genes
    }

    _split_files = {"train": "idx_train.txt", "val": "idx_val.txt", "test": "idx_test.txt"}

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
                download(url, filepath, filedesc)

        need_split = False
        for filename in self._split_files.values():
            filepath = os.path.join(self._data_dir, filename)
            if not os.path.isfile(filepath):
                need_split = True
                break
        if need_split:
            snare_bcs = pd.read_table(
                os.path.join(self._data_dir, self._files["snareseq_barcodes"]), sep="\t", index_col=0
            )
            split = torch.utils.data.random_split(snare_bcs, [0.6, 0.2, 0.2])
            for fname, dset in zip(self._split_files.values(), split, strict=False):
                np.savetxt(os.path.join(self._data_dir, fname), dset.indices, fmt="%d")

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
