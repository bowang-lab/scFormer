# %%
import sys
import argparse
import json
import time
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import scanpy as sc
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

sys.path.insert(0, "../")
import scformer as scf
from scformer.model import TransformerModel
from scformer.tokenizer import tokenize_and_pad_batch, random_mask_value
from scformer import logger

sc.set_figure_params(figsize=(4, 4))
scf.utils.set_seed(42)

# %%
# argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--data-source",
    type=str,
    required=True,
    help='The name of the data source (currently support "scvi" datasets), or the '
    "path to the data file.",
)
parser.add_argument(
    "-m",
    "--model-dir",
    type=str,
    required=True,
    help="The path to the model directory.",
)
parser.add_argument(
    "-s",
    "--save-dir",
    type=str,
    required=True,
    help="The directory to save the trained model and the results.",
)

# settings for evaluation
parser.add_argument(
    "--cell-emb-mode",
    type=str,
    choices=["weighted sum", "non-weighted sum", "cls"],
    default="weighted sum",
    help="The mode to use for cell embeddings.",
)
parser.add_argument(
    "--eval-batch-size",
    type=int,
    default=32,
    help="The batch size for evaluation. Default is 32.",
)

# settings for data
parser.add_argument(
    "--n-hvg",
    type=int,
    default=0,
    help="The number of highly variable genes. If set to 0, will use all genes. "
    "Default is 0, which will determine the n_hvg automatically.",
)

# settings for tokenizer
parser.add_argument(
    "--pad-token",
    type=str,
    default="<pad>",
    help="The token to use for padding. Default is <pad>.",
)
parser.add_argument(
    "--pad-value",
    type=int,
    default=0,
    help="The value to use for padding null gene expression. Default is 0.",
)
parser.add_argument(
    "--max-seq-len",
    type=int,
    default=1024,
    help="The maximum length of the sequence. Default is 1000. The actual used "
    "max length would be the minimum of this value and the length of the longest "
    "sequence in the data.",
)

# settings for logging
parser.add_argument(
    "--log-interval",
    type=int,
    default=100,
    help="The interval for logging. Default is 100.",
)


if scf.utils.isnotebook():
    model_name = "pbmc-Jun09-22-32-2022"
    args = parser.parse_args(
        args=[
            "-d",
            "pbmc_dataset",
            "-m",
            f"./save/{model_name}",
            "-s",
            f"./save/apply-{model_name}",
        ]
    )
else:
    args = parser.parse_args()

# %% settings
print(args)
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

with open(save_dir / "args.json", "w") as f:
    json.dump(vars(args), f, indent=2)
scf.utils.add_file_handler(logger, save_dir / "run.log")

special_tokens = [args.pad_token, "<cls>", "<eoc>"]

# %% [markdown]
# # Load and prepare data
# check if the data source is a file path
if Path(args.data_source).is_file():
    adata = sc.read(args.data_source, cache=True)
    # Specific the required column names, when loading the data the first time.
    # Store the column names for later use.
    (
        celltype_col,
        str_celltype_col,
        gene_col,
        batch_key,
    ) = scf.utils.find_required_colums(
        adata,
        id=args.data_source,
        configs_dir=Path(args.data_source).parent,
    )
elif args.data_source == "pbmc_dataset":
    adata = sc.datasets.pbmc3k_processed()
    adata.obs["batch"] = (adata.obs["n_genes"] > adata.obs["n_genes"].mean()).astype(
        int
    )
    adata.var["gene_symbols"] = adata.var.index
    str_celltype_col = "louvain"
    gene_col = "gene_symbols"
    batch_key = "batch"
    celltype_col = None
if celltype_col is None:
    celltype_col = "int" + str_celltype_col
    adata.obs[celltype_col] = scf.utils.category_str2int(adata.obs[str_celltype_col])

# sc.pp.filter_genes(adata, min_counts=3 / 10000 * adata.n_obs)

adata.layers["counts"] = adata.X.copy()  # preserve counts
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata  # freeze the state in `.raw`

# make the batch category column
if not isinstance(adata.obs[batch_key][0], str):
    adata.obs["str_" + batch_key] = adata.obs[batch_key].astype(str)
    batch_key = "str_" + batch_key

# filter highly variable genes
if args.n_hvg is None or args.n_hvg > 0:
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_hvg,
        subset=True,
        flavor="cell_ranger",  # the cell_rager flaver expects logarithmized data
    )
logger.info(adata)

# %% [markdown]
# # Tokenize input
if isinstance(adata.layers["counts"], np.ndarray):
    all_counts = adata.layers["counts"]
else:
    all_counts = adata.layers["counts"].A

num_of_non_zero_genes = [
    np.count_nonzero(all_counts[i]) for i in range(all_counts.shape[0])
]
max_length = np.max(num_of_non_zero_genes) + 1  # plus 1 for appending <cls>
max_length = min(max_length, args.max_seq_len)

max_value = np.quantile(all_counts[np.nonzero(all_counts)], 0.99)
all_counts = np.clip(all_counts, None, max_value)

# %% [markdown]
# # Load model and vocabulary
model_dir = Path(args.model_dir)
vocab_file = model_dir / "vocab.json"
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"

# vocabulary
with open(vocab_file, "r") as f:
    vocab = json.load(f)

# configs
with open(model_config_file, "r") as f:
    model_configs = json.load(f)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(
    ntoken=len(vocab),
    d_model=model_configs["embsize"],
    nhead=model_configs["nheads"],
    d_hid=model_configs["d_hid"],
    nlayers=model_configs["nlayers"],
    nlayers_cls=model_configs["n_layers_cls"],
    n_cls=1,  # TODO: fix loading this
    vocab=vocab,
    dropout=model_configs["dropout"],
    pad_token=model_configs["pad_token"],
    pad_value=model_configs["pad_value"],
)
model.to(device)
try:
    model.load_state_dict(torch.load(model_file, map_location=device))
except:
    params = model.state_dict()
    for key, value in torch.load(model_file, map_location=device).items():
        # only load params that are in the current model
        if key in model.state_dict() and model.state_dict()[key].shape == value.shape:
            params[key] = value
    model.load_state_dict(params)
model.eval()

# %% [markdown]
# # Gene embeddings
gene_embeddings = model.encoder(torch.arange(len(vocab)).to(device))
gene_embeddings = gene_embeddings.detach().cpu().numpy()

# %%
adata.var["id_in_vocab"] = [vocab.get(gene, -1) for gene in adata.var[gene_col]]
# TODO: replace the hardcoded -1 with proper unkown token handling
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
logger.info(
    f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
    f"in vocabulary of size {len(vocab)}."
)

# %% [markdown]
# # Cell embeddings
def get_batch_cell_embeddings(
    adata,
    gene_embs: np.ndarray,
    count_matrix: np.ndarray = None,
    cell_embedding_mode: str = "weighted sum",
) -> np.ndarray:
    """
    Get the cell embeddings for a batch of cells.

    Args:
        adata (AnnData): The AnnData object.
        gene_embs (np.ndarray): The gene embeddings, shape (len(vocab), d_emb).
        count_matrix (np.ndarray): The count matrix.

    Returns:
        np.ndarray: The cell embeddings.
    """
    if count_matrix is None:
        count_matrix = adata.layers["counts"].A

    assert count_matrix.shape == (len(adata.obs), len(adata.var))

    # gene vocabulary ids
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    in_vocab_mask = gene_ids_in_vocab >= 0
    matched_gene_ids_in_vocab = gene_ids_in_vocab[in_vocab_mask]

    matched_gene_embs = gene_embs[
        matched_gene_ids_in_vocab, :
    ]  # (n_matched_genes, d_emb)
    matched_cell_counts = count_matrix[:, in_vocab_mask]  # (n_cells, n_matched_genes)

    if cell_embedding_mode == "weighted sum":
        cell_embeddings = np.matmul(matched_cell_counts, matched_gene_embs)
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
    elif cell_embedding_mode == "non-weighted sum":
        cell_embeddings = np.matmul(matched_cell_counts > 0, matched_gene_embs)
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
    elif cell_embedding_mode == "cls":
        tokenized_all = tokenize_and_pad_batch(
            all_counts[:, in_vocab_mask],
            matched_gene_ids_in_vocab,
            max_len=max_length,  # TODO: try to use all genes for the cls embedding
            vocab=vocab,
            pad_token=model_configs["pad_token"],
            pad_value=model_configs["pad_value"],
            append_cls=True,  # append <cls> token at the beginning
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[model_configs["pad_token"]])
        with torch.no_grad():
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_size=args.eval_batch_size,
            )[:, 0, :]
            cell_embeddings = cell_embeddings.cpu().numpy()
    else:
        raise ValueError(f"Unknown cell embedding mode: {cell_embedding_mode}")
    return cell_embeddings


# %%
cell_embeddings = get_batch_cell_embeddings(
    adata,
    gene_embeddings,
    count_matrix=all_counts,
    cell_embedding_mode=args.cell_emb_mode,
)
adata.obsm["X_scFormer"] = cell_embeddings

# %% visualization
sc.pp.neighbors(adata, use_rep="X_scFormer")
sc.tl.umap(adata, min_dist=0.3)

fig = sc.pl.umap(
    adata,
    color=[batch_key, str_celltype_col],
    ncols=2,
    frameon=False,
    return_fig=True,
)
fig.savefig(
    save_dir / f"embeddings_umap[{args.cell_emb_mode}].png",
    bbox_inches="tight",
)

# %% scib metrics
import scib

# Wrapper for all scib metrics, we leave out some metrics like hvg_score, cell_cyvle,
# trajectory_conservation, because we only evaluate the latent embeddings here and
# these metrics are evaluating the reconstructed gene expressions or pseudotimes.
results = scib.metrics.metrics(
    adata,
    adata_int=adata,
    batch_key=batch_key,
    label_key=str_celltype_col,
    embed="X_scFormer",
    isolated_labels_asw_=False,
    silhouette_=False,
    hvg_score_=False,
    graph_conn_=True,
    pcr_=False,
    isolated_labels_f1_=False,
    trajectory_=False,
    nmi_=True,  # use the clustering, bias to the best matching
    ari_=True,  # use the clustering, bias to the best matching
    cell_cycle_=False,
    kBET_=False,  # kBET return nan sometimes, need to examine
    ilisi_=False,
    clisi_=False,
)

logger.info(f"{results}")

results = results[0].dropna().to_dict()

with open(save_dir / "results.json", "w") as f:
    json.dump(results, f)
