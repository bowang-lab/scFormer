# %%
import json
import os
import time
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import torch
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

import scformer as scf
from scformer.model import TransformerGenerator
from scformer.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scformer.utils import set_seed, category_str2int

matplotlib.rcParams["savefig.transparent"] = False

set_seed(42)

# %% [markdown]
# ## Training Settings

# %%
# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 2

n_hvg = 0  # number of highly variable genes
include_zero_gene = "batch-wise"  # include zero expr genes in training input, "batch-wise", "row-wise", or False
max_seq_len = 1536

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
cell_emb_style = "cls"
mvc_decoder_style = "inner product, detach"
ecs_threshold = 0.85

# settings for optimizer
lr = 1e-4  # or 1e-4
batch_size = 32
eval_batch_size = 32  # FIXME: how come eval_batch_size changes the pert plot? fix this
epochs = 6
schedule_interval = 1

# settings for the model
embsize = 64  # embedding dimension
d_hid = 64  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability

# logging
log_interval = 100

# dataset and evaluation choices
data_name = "norman"
split = "simulation"
if data_name == "norman":
    perts_to_plot = ["SAMD1+ZBTB1"]
elif data_name == "adamson":
    perts_to_plot = ["KCTD16+ctrl"]
elif data_name == "dixit":
    perts_to_plot = ["ELK1+ctrl"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
save_dir = Path(
    f"./save/dev_perturb_{data_name}"
    f"{'_cce' if CCE else ''}{'_cls' if CLS else ''}"
    f"{'_mvc' if MVC else ''}{'_ecs' if ECS else ''}"
    f"{'_hvg' if n_hvg > 0 else ''}"
    f"-{time.strftime('%b%d-%H-%M')}/"
)
save_dir.mkdir(parents=True, exist_ok=True)
print(f"saving to {save_dir}")
# save the whole script to the dir
os.system(f"cp {__file__} {save_dir}")

logger = scf.logger
scf.utils.add_file_handler(logger, save_dir / "run.log")


# %%
pert_data = PertData("./data")
pert_data.load(data_name=data_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

# %% [markdown]
# ### DATA Specifications
# Each row/observation in adata is cell expression profile, and the obs["condition"] is the condition. Using the condition to determine whether the row goes to train, valid, or test.

# %%
# batch_data = iter(pert_data.dataloader["train_loader"]).next()
# batch_data

# %%
# find max non_zeros_gene_idx
length = []
for k, gene_idx in pert_data.adata.uns["non_zeros_gene_idx"].items():
    length.append(gene_idx.shape[0])
print(f"max num of non_zeros_gene_idx: {max(length)}")
print(f"min num of non_zeros_gene_idx: {min(length)}")
print(f"mean num of non_zeros_gene_idx: {np.mean(length)}")

# %% [markdown]
# ## Simple training using the GEARS dataloader
# where the target cell is simply randomly chosen from the preturbed cell types.
#
# Next step, using our own data processing and training manner.

# %%
genes = pert_data.adata.var["gene_name"].tolist()
vocab = Vocab(
    VocabPybind(genes + special_tokens, None)
)  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)
n_genes = len(genes)


# %% [markdown]
# # Create and train scFormer

# %% The default model takes 1.36 GB in this cell.
ntokens = len(vocab)  # size of vocabulary
model = TransformerGenerator(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    pert_pad_id=pert_pad_id,
    do_mvc=MVC,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
).to(device)


# %%
def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()


def tensorlist2tensor(tensorlist, pad_value=pad_value):
    max_len = max(len(t) for t in tensorlist)
    dtype = tensorlist[0].dtype
    device = tensorlist[0].device
    tensor = torch.zeros(len(tensorlist), max_len, dtype=dtype, device=device)
    tensor.fill_(pad_value)
    for i, t in enumerate(tensorlist):
        tensor[i, : len(t)] = t
    return tensor


criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse, total_cls, total_cce, total_mvc, total_ecs = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    total_error = 0.0
    input_zero_ratios = []
    target_zero_ratios = []
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if include_zero_gene == "batch-wise":
            input_gene_ids = (
                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            )
            if len(input_gene_ids) > max_seq_len:
                # sample input_gene_id
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]
            input_gene_ids = input_gene_ids.repeat(batch_size, 1)

            input_zero_ratios.append(
                torch.sum(input_values == 0).item() / input_values.numel()
            )
            target_zero_ratios.append(
                torch.sum(target_values == 0).item() / target_values.numel()
            )

        else:
            input_gene_ids = []
            input_values = []
            input_pert_flags = []
            target_values = []
            # TODO: should make sure at least the pert flag gene is included
            if include_zero_gene == "row-wise":
                tmp_ = torch.logical_or(ori_gene_values != 0, target_gene_values != 0)
            elif not include_zero_gene:
                tmp_ = ori_gene_values != 0
            else:
                raise ValueError(
                    "include_zero_gene must be one of batch-wise, row-wise, or False."
                )
            for row_i in range(batch_size):
                input_gene_id = tmp_[row_i].nonzero().flatten()
                if len(input_gene_id) > max_seq_len:
                    # sample input_gene_id
                    input_gene_id = torch.randperm(len(input_gene_id), device=device)[
                        :max_seq_len
                    ]
                input_gene_ids.append(input_gene_id)
                input_values.append(ori_gene_values[row_i][input_gene_id])
                input_pert_flags.append(pert_flags[row_i][input_gene_id])
                target_values.append(target_gene_values[row_i][input_gene_id])
            input_gene_ids = tensorlist2tensor(
                input_gene_ids, pad_value=vocab[pad_token]
            )
            input_values = tensorlist2tensor(input_values)
            input_pert_flags = tensorlist2tensor(
                input_pert_flags, pad_value=pert_pad_id
            )
            target_values = tensorlist2tensor(target_values)

            input_zero_ratios.append(
                torch.sum(
                    torch.logical_and(
                        input_values == 0, input_gene_ids != vocab[pad_token]
                    )
                ).item()
                / torch.sum(input_gene_ids != vocab[pad_token]).item()
            )
            target_zero_ratios.append(
                torch.sum(
                    torch.logical_and(
                        target_values == 0, input_gene_ids != vocab[pad_token]
                    )
                ).item()
                / torch.sum(input_gene_ids != vocab[pad_token]).item()
            )
        # TODO: how do you deal with the zeros in target? Should you ignore them
        # in the objective computation, or at least use a smaller weight? Otherwise,
        # the model will be encouraged to predict small values in general.
        # - In self-supervised settings. The target zeros may have more sense to
        #   learn to. Since zeros likely mean small gene expression.
        # - In at learst the perturbation setting, it is not clear the zeros make
        #   any sense. Since that's input and target cell pairs, and at least in
        #   the input, the specific gene is well expressed. It sometimes can be
        #   just not very well paired and the target cell behaves differently.

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        output_dict = model(
            input_gene_ids,
            input_values,
            input_pert_flags,
            src_key_padding_mask=src_key_padding_mask,
            CLS=CLS,
            CCE=CCE,
            MVC=MVC,
            ECS=ECS,
        )
        output_values = output_dict["mlm_output"]

        masked_positions = torch.ones_like(input_values, dtype=torch.bool)  # Use all
        loss = loss_mse = criterion(output_values, target_values, masked_positions)
        if CLS:
            target_labels = torch.tensor(train_labels[i : i + batch_size]).to(device)
            loss_cls = criterion_cls(output_dict["cls_output"], target_labels)
            loss = loss + loss_cls
        if CCE:
            loss_cce = 10 * output_dict["loss_cce"]
            loss = loss + loss_cce
        if MVC:
            loss_mvc = criterion(
                output_dict["mvc_output"], target_values, masked_positions
            )
            loss = loss + loss_mvc
        if ECS:
            loss_ecs = 10 * output_dict["loss_ecs"]
            loss = loss + loss_ecs

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            mre = masked_relative_error(output_values, target_values, masked_positions)

        # torch.cuda.empty_cache()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_cls += loss_cls.item() if CLS else 0.0
        total_cce += loss_cce.item() if CCE else 0.0
        total_mvc += loss_mvc.item() if MVC else 0.0
        total_ecs += loss_ecs.item() if ECS else 0.0
        total_error += mre.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if CLS else 0.0
            cur_cce = total_cce / log_interval if CCE else 0.0
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_ecs = total_ecs / log_interval if ECS else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"cls {cur_cls:5.2f} | " if CLS else "")
                + (f"cce {cur_cce:5.2f} |" if CCE else "")
                + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
                + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
            )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_error = 0
            start_time = time.time()

    input_zero_ratios = np.array(input_zero_ratios)
    logger.info(
        f"input_zero_ratios: {input_zero_ratios.mean():.3f} +- {input_zero_ratios.std():.3f}"
    )
    target_zero_ratios = np.array(target_zero_ratios)
    logger.info(
        f"target_zero_ratios: {target_zero_ratios.mean():.3f} +- {target_zero_ratios.std():.3f}"
    )


def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            pert_flags = x[:, 1].long().view(batch_size, n_genes)
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            if include_zero_gene == "batch-wise":
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
                if len(input_gene_ids) > max_seq_len:
                    # sample input_gene_id
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :max_seq_len
                    ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]
                input_gene_ids = input_gene_ids.repeat(batch_size, 1)

            else:
                input_gene_ids = []
                input_values = []
                input_pert_flags = []
                target_values = []
                if include_zero_gene == "row-wise":
                    tmp_ = torch.logical_or(
                        ori_gene_values != 0, target_gene_values != 0
                    )
                elif not include_zero_gene:
                    tmp_ = ori_gene_values != 0
                else:
                    raise ValueError(
                        "include_zero_gene must be one of 'batch-wise', 'row-wise', False"
                    )
                for row_i in range(batch_size):
                    # nonzero indices of the gene values
                    input_gene_id = tmp_[row_i].nonzero().flatten()
                    input_gene_ids.append(input_gene_id)
                    input_values.append(ori_gene_values[row_i][input_gene_id])
                    input_pert_flags.append(pert_flags[row_i][input_gene_id])
                    target_values.append(target_gene_values[row_i][input_gene_id])
                input_gene_ids = tensorlist2tensor(
                    input_gene_ids, pad_value=vocab[pad_token]
                )
                input_values = tensorlist2tensor(input_values)
                input_pert_flags = tensorlist2tensor(
                    input_pert_flags, pad_value=pert_pad_id
                )
                target_values = tensorlist2tensor(target_values)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            output_dict = model(
                input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool, device=input_values.device
            )
            loss = criterion(output_values, target_values, masked_positions)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item()
    return total_loss / len(val_loader), total_error / len(val_loader)


# %%
best_val_loss = float("inf")
best_model = None


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loader = pert_data.dataloader["train_loader"]
    valid_loader = pert_data.dataloader["val_loader"]

    train(
        model,
        train_loader,
    )
    val_loss, val_mre = evaluate(
        model,
        valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    scheduler.step()


# %% save the best model
torch.save(best_model.state_dict(), save_dir / "best_model.pt")

# %% [markdown]
# ## Evaluations

# %%
def predict(
    model: TransformerGenerator, pert_list: List[str], pool_size: Optional[int] = None
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()
    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(batch_data, include_zero_gene)
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    return results_pred


# %%
def plot_perturbation(
    model: nn.Module, query: str, save_file: str = None, pool_size: int = None
):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    adata = pert_data.adata
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    genes = [
        gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
    if query.split("+")[1] == "ctrl":
        pred = predict(model, [[query.split("+")[0]]], pool_size=pool_size)
        pred = pred[query.split("+")[0]][de_idx]
    else:
        pred = predict(model, [query.split("+")], pool_size=pool_size)
        pred = pred["_".join(query.split("+"))][de_idx]
    ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values

    pred = pred - ctrl_means
    truth = truth - ctrl_means

    plt.figure(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

    for i in range(pred.shape[0]):
        _ = plt.scatter(i + 1, pred[i], color="red")

    plt.axhline(0, linestyle="dashed", color="green")

    ax = plt.gca()
    ax.xaxis.set_ticklabels(genes, rotation=90)

    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine()

    if save_file:
        plt.savefig(save_file, bbox_inches="tight", transparent=False)
    plt.show()


# %%
# predict(best_model, [["FEV"], ["FEV", "SAMD11"]])
for p in perts_to_plot:
    plot_perturbation(best_model, p, pool_size=300, save_file=f"{save_dir}/{p}.png")

# %%
def eval_perturb(
    loader: DataLoader, model: TransformerGenerator, device: torch.device
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):

        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(batch, include_zero_gene)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float)

    return results


# %%
test_loader = pert_data.dataloader["test_loader"]
test_res = eval_perturb(test_loader, best_model, device)
test_metrics, test_pert_res = compute_metrics(test_res)
print(test_metrics)
# NOTE: mse and pearson corr here are computed for the mean pred expressions vs. the
# truth mean across all genes. Further, one can compute the distance of two distributions.

# save the dicts in json
with open(f"{save_dir}/test_metrics.json", "w") as f:
    json.dump(test_metrics, f)
with open(f"{save_dir}/test_pert_res.json", "w") as f:
    json.dump(test_pert_res, f)

deeper_res = deeper_analysis(pert_data.adata, test_res)
non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

metrics = ["pearson_delta", "pearson_delta_de"]
metrics_non_dropout = [
    "frac_correct_direction_top20_non_dropout",
    "frac_correct_direction_non_dropout",
    "pearson_delta_top20_de_non_dropout",
    "pearson_top20_de_non_dropout",
]
subgroup_analysis = {}
for name in pert_data.subgroup["test_subgroup"].keys():
    subgroup_analysis[name] = {}
    for m in metrics:
        subgroup_analysis[name][m] = []

    for m in metrics_non_dropout:
        subgroup_analysis[name][m] = []

for name, pert_list in pert_data.subgroup["test_subgroup"].items():
    for pert in pert_list:
        for m in metrics:
            subgroup_analysis[name][m].append(deeper_res[pert][m])

        for m in metrics_non_dropout:
            subgroup_analysis[name][m].append(non_dropout_res[pert][m])

for name, result in subgroup_analysis.items():
    for m in result.keys():
        subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
        logger.info("test_" + name + "_" + m + ": " + str(subgroup_analysis[name][m]))

# %%
for p in np.unique(np.array([item.pert for item in test_loader.dataset]))[:30]:
    p = str(p)
    plot_perturbation(
        best_model,
        p,
        save_file=f"{save_dir}/test_{p}.png",
        pool_size=300,
    )

# %% [markdown]
# ### Correlation plots
