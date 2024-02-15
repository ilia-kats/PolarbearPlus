import gzip
import os

import numpy as np
import pandas as pd
import requests
import torch
from gtfparse import read_gtf
from lightning import pytorch as pl
from scipy.io import mmread
from torch.utils.data import DataLoader, random_split


class SNAREseq:
    def __init__(self, root="/data/mikulik", data_dir="snare", modality="rna", download=False):
        self.root = root  # where to save the data
        self.data_dir = data_dir  # create new directory called "snare" in root
        self.files = [
            "cDNA.barcodes.tsv.gz",
            "cDNA.counts.mtx.gz",
            "cDNA.genes.tsv.gz",
            "chromatin.barcodes.tsv.gz",
            "chromatin.counts.mtx.gz",
            "chromatin.peaks.tsv.gz",
        ]
        self.base_url = (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE126nnn/GSE126074/suppl/GSE126074%5FAdBrainCortex%5FSNAREseq%5F"
        )
        self.download = download
        self.modality = modality  # rna or atac
        self.logbatchmean = None
        self.logbatchvar = None
        if self.download:
            self.download_data()
        self.load_data()

        # count matrix, cell barcodes
        self.data, self.cells = self.prepare_data()

    def download_data(self):
        if os.path.isdir(os.path.join(self.root, self.data_dir)):
            print("Files already downloaded")
            return
        else:
            os.makedirs(os.path.join(self.root, self.data_dir))
            for filename in self.files:
                url = f"{self.base_url}{filename}"
                try:
                    print(f"Downloading {url}")
                    r = requests.get(url, timeout=10)
                    with open(os.path.join(self.root, self.data_dir, filename), "wb") as f:
                        f.write(r.content)

                except requests.exceptions.RequestException:
                    print(f"Failed to download {url}")
            # Download the genome annotations
            r = requests.get(
                "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M10/gencode.vM10.annotation.gtf.gz",
                timeout=10,
            )
            with open(os.path.join(self.root, self.data_dir, "gencode.vM10.annotation.gtf.gz"), "wb") as f:
                f.write(r.content)

    def load_data(self):
        with gzip.open(os.path.join(self.root, self.data_dir, "gencode.vM10.annotation.gtf.gz")) as f:
            gene_anno = read_gtf(f)
        self.gene_anno = pd.DataFrame(gene_anno, columns=gene_anno.columns)

        for file in self.files:
            with gzip.open(os.path.join(self.root, self.data_dir, file)) as f:
                if file.startswith("cDNA.genes") & file.endswith("tsv.gz"):
                    self.genes = pd.read_csv(f, sep="\t", header=None)
                elif file.startswith("cDNA") & file.endswith("mtx.gz"):
                    self.gex = mmread(f)
                elif file.startswith("cDNA.barcodes") & file.endswith("tsv.gz"):
                    self.gex_cells = pd.read_csv(f, sep="\t", header=None)
                elif file.startswith("chromatin.peaks") & file.endswith("tsv.gz"):
                    self.peaks = pd.read_csv(f, sep="\t", header=None)
                elif file.startswith("chromatin.barcodes") & file.endswith("tsv.gz"):
                    self.atac_cells = pd.read_csv(f, sep="\t", header=None)
                elif file.startswith("chromatin") & file.endswith("mtx.gz"):
                    self.atac = mmread(f)
        print("All files loaded")

    def prepare_data(self):
        genes_keep = self.genes.iloc[
            np.where(((self.gex > 0).sum(axis=1) > 200) | ((self.gex > 0).sum(axis=1) > 2500))[0]
        ]

        # add gene annotatioins
        genes_keep = genes_keep[genes_keep[0].isin(self.gene_anno.gene_name)]
        genes_keep.columns = ["gene_name"]
        gene_anno = self.gene_anno[self.gene_anno["gene_name"].isin(genes_keep.gene_name)]
        gene_anno = gene_anno.drop_duplicates(subset="gene_name", keep="first")
        genes_keep = genes_keep.merge(gene_anno, how="left", on="gene_name")
        self.gex = self.gex.toarray()[genes_keep.index, :]

        mask = self.atac > 0
        perc = mask.sum(axis=1) / mask.shape[1]
        rowsum = mask.sum(axis=1)
        # keep peaks that occur in more than 5 cells and less than 10% of cells
        keep = np.intersect1d(np.where(rowsum > 5)[0], np.where(perc < 0.1)[0])
        self.peaks = self.peaks.iloc[keep]
        self.atac = self.atac.toarray()[keep, :]

        # cell barcodes
        a, b = self.atac_cells.reset_index(), self.gex_cells.reset_index()
        a.columns, b.columns = ["index_atac", "cell"], ["index_gex", "cell"]
        self.barcodes = a.merge(b, how="left", on="cell")

        # reorder gene expression matrix
        self.gex = self.gex[:, self.barcodes.index_gex]
        self.atac = self.atac[:, self.barcodes.index_atac]

        if self.modality == "atac":
            return self.atac.T, self.atac_cells[0]
        elif self.modality == "rna":
            return self.gex.T, self.atac_cells[0]
        else:
            print("Modality not found")
            return None

    def __len__(self):
        return len(self.atac_cells)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx, :], self.cells[idx]


class SNAREDataModule(pl.LightningDataModule):
    def __init__(self, root, data_dir, modality="rna", batch_size: int = 32):
        super().__init__()
        self.root = root
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.modality = modality

    def setup(self, stage: str):
        snare = SNAREseq(self.root, self.data_dir, download=True, modality=self.modality)
        self.snare_train, self.snare_val, self.snare_test = random_split(
            snare, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.snare_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.snare_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.snare_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #    return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        print(stage)  # otherwise git hooks will fail
        ...
