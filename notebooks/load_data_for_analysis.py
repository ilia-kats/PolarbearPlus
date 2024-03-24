import os

import numpy as np
import pandas as pd
from scipy.io import mmread

def load_all_data(data_dir):
    """
    Load all.
    """
    snareseq_barcodes, snareseq_peaks, gene_names = load_snare_data(os.path.join(data_dir, "snareseq"))
    true_split = get_test_split_index(os.path.join(data_dir, "split"))
    test_cell_names = get_celltype_annotations(os.path.join(data_dir, "diffexp"), snareseq_barcodes=snareseq_barcodes, true_split=true_split)

    diff_genes = load_diff_genes(os.path.join(data_dir, "diffexp"))
    diff_genes_ind = get_diff_gene_ind(gene_names, diff_genes)

    rna_counts, obs_rna = get_rna_counts(os.path.join(data_dir,"snareseq/"), true_split)
    atac_counts, obs_atac = get_atac_counts(os.path.join(data_dir,"snareseq/"), true_split)

    diff_peaks_ind = get_diff_peaks_ind(os.path.join(data_dir, "diffexp"), snareseq_peaks)
    return snareseq_barcodes, snareseq_peaks, gene_names, true_split, test_cell_names, diff_genes, diff_genes_ind, rna_counts, obs_rna, atac_counts, obs_atac, diff_peaks_ind


def load_snare_data(data_dir):
    """
    Loading Gene names, peak names, cell barcodes. 
    """
    snareseq_barcodes = pd.read_csv(
        os.path.join(data_dir, "adultbrainfull50_atac_outer_snareseq_barcodes.tsv"), sep="\t"
    )
    #snareseq_genes = pd.read_csv(os.path.join(data_dir, "adultbrainfull50_rna_outer_genes.txt"), sep="\t")
    snareseq_peaks = pd.read_csv(os.path.join(data_dir, "adultbrainfull50_atac_outer_peaks.txt"), sep="\t", header=None)
    snareseq_peaks.columns = ["peaks"]
    snareseq_peaks["ind"] = snareseq_peaks.index
    snare_seq_genes = pd.read_csv(os.path.join(data_dir, "adultbrainfull50_rna_outer_genes.txt"), sep="\t", header=None)
    return snareseq_barcodes, snareseq_peaks, snare_seq_genes


def load_diff_genes(data_dir):
    """
    Loading differentially expressed Gene names. 
    """
    diff_genes = pd.read_excel(
        os.path.join(data_dir, "NIHMS1539957-supplement-sup_tab1.xlsx"), sheet_name="Adult_cerebral_cortex", header=3
    )
    return diff_genes


def get_diff_gene_ind(gene_names, diff_genes):
    """
    Get index of differentially expressed genes.
    """
    gene_names.columns = ["gene"]
    gene_names["ind"] = np.arange(len(gene_names))
    diff_genes_ind = gene_names[gene_names.gene.isin(diff_genes.Gene.unique())]["ind"]
    return diff_genes_ind


def get_test_split_index(data_dir):
    """
    Get cell indices of test split. 
    """
    true_split = pd.read_csv(os.path.join(data_dir, "idx_test.txt"), sep="\t", header=None)
    true_split.columns = ["idx"]
    return true_split


def get_celltype_annotations(data_dir, snareseq_barcodes, true_split, file_name="snareseq_anno.csv"):
    """
    Get cell type annotations. 
    """
    # get cell type annotations from snareseq
    cell_names = pd.read_csv(os.path.join(data_dir, "snareseq_anno.csv"), index_col=0)
    cell_names = cell_names.loc[snareseq_barcodes["index"]]
    # change order of cell names so they comply with the test set
    test_cell_names = cell_names.iloc[true_split.idx.tolist()]
    test_cell_names.Ident.unique()
    return test_cell_names


def get_rna_counts(data_dir, true_split):
    """
    RNA count matrix. 
    """
    # Read observed RNA and ATAC counts
    rna_counts = (
        mmread(os.path.join(data_dir, "adultbrainfull50_rna_outer_snareseq.mtx"))
        .tocsr()
        .astype(np.float32)
        .toarray()
        .squeeze()
    )
    # subset to keep only test set cell
    obs_rna = rna_counts[true_split.idx.tolist(), :]
    return rna_counts, obs_rna


def get_atac_counts(data_dir, true_split):
    """
    ATAC count matrix. 
    """
    atac_counts = mmread(os.path.join(data_dir, "adultbrainfull50_atac_outer_snareseq.mtx")).tocsr().astype(np.float32)
    atac_counts.data[:] = 1  # set all non-zero values to 1
    atac_counts = atac_counts.toarray()  # .squeeze()
    # subset to keep only test set cells
    obs_atac = atac_counts[true_split.idx.tolist(), :]
    return atac_counts, obs_atac


def get_diff_peaks_ind(data_dir, snareseq_peaks):
    """
    Get differentially accessible peak indices. 
    """
    sheet_names = [
        "Ast",
        "Claustrum",
        "Ex-L23-Rasgrf2",
        "Ex-L34-Rmst",
        "Ex-L34-Rorb",
        "Ex-L45-Il1rapl2",
        "Ex-L45-Thsd7a",
        "Ex-L5-Galnt14",
        "Ex-L5-Parm1",
        "Ex-L56-Sulf1",
        "Ex-L56-Tshz2",
        "Ex-L6-Tle4",
        "In-Npy",
        "In-Pvalb",
        "In-Sst",
        "In-Vip",
        "Oli-Itpr2",
        "Oli-Mal",
        "OPC",
        "Endo",
        "Peri",
        "Mic",
    ]

    diff_peak_list = []
    for i in sheet_names:
        if i == "Ast":
            diff_peaks = pd.read_excel(
                os.path.join(data_dir, "NIHMS1539957-supplement-sup_tab3.xlsx"), sheet_name=i, header=3
            )
        else:
            diff_peaks = pd.read_excel(
                os.path.join(data_dir, "NIHMS1539957-supplement-sup_tab3.xlsx"), sheet_name=i, header=0
            )
        diff_peaks["peaks"] = (
            diff_peaks.chrom.astype("str") + ":" + diff_peaks.start.astype("str") + "-" + diff_peaks.end.astype("str")
        )
        diff_peak_list = diff_peak_list + diff_peaks.peaks.tolist()
    # print(len(diff_peak_list))
    # print(f"Unique: {len(set(diff_peak_list))}")

    diff_peaks_ind = snareseq_peaks[snareseq_peaks.peaks.isin(diff_peak_list)].ind.tolist()
    return diff_peaks_ind
