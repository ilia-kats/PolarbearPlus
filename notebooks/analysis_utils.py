import pandas as pd
from scipy.io import mmread
import os
import numpy as np
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import zarr
import os
import torch
import anndata
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def compare_correlations(x, y, title="", x_label="", y_label="", save_fig=False, file_name="corr.pdf"):
    """
    Scatterplot of correlations.
    """
    sns.scatterplot(x=x, y=y, s=15, c="black")
    plt.title(title, fontsize=30)
    sns.lineplot(x=x, y=x, c="black")
    plt.ylabel(f"{y_label}, mean:{y.mean():.3f}" , fontsize=20)
    plt.xlabel(f"{x_label}, mean: {x.mean():.3f}", fontsize=20)
    plt.xticks(fontsize=15)#, rotation=90)
    plt.yticks(fontsize=15)#, rotation=90)
    sns.kdeplot(x=x, y=y, fill=True, alpha=0.7, cmap="Blues")
    if save_fig is not None:
        plt.savefig(os.path.join(save_fig, file_name))
        plt.show()

def corr_histograms(dict, labels, colors, save_figs=None, file_name=None, alpha=0.5, bins=100, title="Gene-wise Pearson Correlation", xlabel="correlation"):
    """
    Plot histogram of correlations.
    """
    sns.histplot(data=pd.DataFrame(dict), alpha=alpha, bins=bins, legend=True)#, hue=)
    for i, (name, arr) in enumerate(dict.items()):
        plt.axvline(np.array(arr).mean(), c=colors[i])
    plt.title(title, fontsize=15)
    plt.xlabel(f"{xlabel}", fontsize=15)
    plt.ylabel(None)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if save_figs is not None:
        plt.savefig(os.path.join(save_figs, f"{file_name}.pdf"))
    plt.show()



def normalize_counts(x, normlib=True, logscale=True):
    """
    Library size normalization and log(1+x) transformation.
    """
    if normlib:
        ## compare with normalized true profile
        lib = x.sum(axis=1, keepdims=True)
        x = x / lib
    if logscale:
        x = np.log1p(x)
    return x

def corr_obs_reocnstr(x, y, per_gene=True, data_type="RNA", plot=False):
    """
    compute correlations across cells or genes. 
    """
    num_genes, num_cells = x.shape[1], y.shape[0]

    x, y = torch.from_numpy(x), torch.from_numpy(y)

    if per_gene:
        pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_genes)
        corr = pearson(x, y)
        if plot:
            sns.histplot(corr)
            if data_type == "RNA":
                plt.title("Correlation across cells for each gene")
            else:
                plt.title("Correlation across cells for each peak")
            plt.xlabel(f"mean: {corr.mean():.3f}")
            plt.show()

    if per_gene == False:
        pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_cells)
        corr = pearson(x.T, y.T)
        if plot:
            sns.histplot(corr)
            if data_type == "RNA":
                plt.title("Correlation across genes for each cell")
            else:
                plt.title("Correlation across peaks for each cell")
            plt.xlabel(f"mean: {corr.mean():.3f}")
            plt.show()
    return corr

def scanpy_embedding(counts, test_cell_names, gene_names, title="", latent=None, return_adata=False, file_name=None, legend_loc=None):
    """
    Use Scanpy and Anndata for UMAP embeddings.
    """
    adata = anndata.AnnData(np.array(counts))
    adata.obs = test_cell_names
    adata.var = gene_names
    if latent is not None:
        adata.obsm["latent_polar"] = latent

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata, min_dist=0.1, n_components=2)
    if file_name is not None:
        sc.pl.umap(adata, color="Ident", title=f"{title}",legend_loc=legend_loc,
                save=f"counts_{file_name}.pdf")
    else:
        sc.pl.umap(adata, color="Ident", title=f"{title}", legend_loc=legend_loc)

    if latent is not None:
        sc.pp.neighbors(adata, n_neighbors=15,  use_rep="latent_polar")
        sc.tl.umap(adata, min_dist=0.1, n_components=2,)
        if file_name is not None:
            sc.pl.umap(adata, color="Ident", title=f"Latent embedding, {title}", legend_loc=legend_loc,
                    save=f"latent_{file_name}.pdf")
        else:
            sc.pl.umap(adata, color="Ident", title=f"Latent embedding, {title}", legend_loc=legend_loc)
    return adata


# Load predictions 

def load_latent_embeddings(zarr_file_path):#
    """
    Load the latent embeddings of models. 
    """
    x = zarr.open(os.path.join(zarr_file_path, "prediction/stats.zarr"), mode="r")
    latent = np.asarray(x["latent_0"])
    return latent

def get_reconstructed_counts(zarr_file):
    """
    Load reconstructed counts of RNA/ATAC VAE.
    """
    zarr_file = os.path.join(zarr_file, "prediction/stats.zarr")
    a = zarr.open(zarr_file, mode="r")
    return a["reconstruction_stats"]["mean"], a["reconstruction_stats"]["var"]

def get_reconstructed_cross_modality(zarr_file):
    """
    Load reconstructed counts of translator networks.
    """
    zarr_file = os.path.join(zarr_file, "prediction/stats.zarr")
    a = zarr.load(zarr_file)
    return a["mean"], a["var"]






#### Confidence intervals
def index_1d_to_2d(index, width):
    """
    Convert 1d to 2d array.
    """
    row = index // width
    col = index % width
    return row, col

def get_index_array(x):
    """
    Create Array of indices.
    """
    ind_arr = np.arange(x.flatten().shape[0]).reshape(x.shape)
    return ind_arr

def scale_quantiles(counts, q025, q975):
    """
    Convert counts using quantiles.
    """
    condition = counts > 0
    obs_scale = (counts[condition] - q025[condition]) / (q975[condition] - q025[condition])
    return obs_scale

def read_vae_quantiles(data_dir, vae="rnavae"):
    """
    Load cVAE quantile preidctions.
    """
    x = zarr.open(os.path.join(data_dir, vae, "settings_polarbear/prediction/stats.zarr"), mode="r")
    q025 = np.asarray(x["reconstruction_stats"]["q0.025"])
    q975 = np.asarray(x["reconstruction_stats"]["q0.975"])
    #obs_norm =obs_rna/obs_rna.sum(axis=1, keepdims=True)
    #print(q025.min(), q025.max())
    return q025, q975


def read_translator_quantiles(data_dir, translator, file):
    """
    Load translator quantile preidctions.
    """
    x = zarr.open(os.path.join(data_dir, translator, file, "prediction/stats.zarr"))
    q025, q95, q975 = np.asarray(x["q0.025"]), np.asarray(x["q0.95"]), np.asarray(x["q0.975"])
    return q025, q975

def plot_ci(obs_scale, title=""):
    """
    Plot quantile scaled counts.
    """
    sns.histplot(obs_scale)
    plt.axvline(1, color="black")
    plt.axvline(0, color="black")
    plt.ylabel("Counts", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(f"{title}", fontsize=30)
    plt.show()



# AUROC

def compute_per_peak_auroc(obs, pred, diff_peaks_ind):
    """
    Computing per peak AUROC.
    """
    # keep only the differentially accessible peaks
    obs = obs[:, diff_peaks_ind]
    pred = pred[:, diff_peaks_ind]

    # remove all peaks for which test set observed counts are all zero counts
    pred = pred[:,  obs.sum(axis=0) != 0]
    obs = obs[:, obs.sum(axis=0) != 0]

    n_peaks=obs.shape[1]
    all_peaks = []
    for peak in range(n_peaks):
        all_peaks.append(roc_auc_score(obs[:, peak], pred[:, peak]))
    return all_peaks

def compute_aupr_norm(obs, pred):
    """
    Normalized AUPR.
    """
    x = average_precision_score(obs, pred)
    pp = np.sum(obs) # number of cells with peak accessible
    pp = pp/obs.shape[0] # number of all cells
    aupr_norm = (x - pp) / (1-pp)
    return aupr_norm

def compute_per_peak_aupr_norm(obs, pred, diff_peaks_ind):
    """
    Computing per peak AUPRnorm.
    """
    # keep only the differentially accessible peaks
    obs = obs[:, diff_peaks_ind] 
    pred = pred[:, diff_peaks_ind]

    # remove all peaks for which test set observed counts are all zero counts
    pred = pred[:,  obs.sum(axis=0) != 0]
    obs = obs[:, obs.sum(axis=0) != 0]

    n_peaks = obs.shape[1]
    all_peaks = []
    for peak in range(n_peaks):
        aupr_norm = compute_aupr_norm(obs[:, peak], pred[:, peak])
        all_peaks.append(aupr_norm)
    return all_peaks
