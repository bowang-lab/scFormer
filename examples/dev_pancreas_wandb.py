# %%
import gc
import os
import time
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
from unittest import loader

import torch
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

import scformer as scf
from scformer.model import TransformerModel, AdversarialDiscriminator
from scformer.tokenizer import tokenize_and_pad_batch, random_mask_value
from scformer.preprocess import Preprocessor
from scformer import SubsetsBatchSampler
from scformer.utils import set_seed, category_str2int

import wandb

sc.set_figure_params(figsize=(4, 4))
set_seed(42)
torch.autograd.set_detect_anomaly(True)

hyperparameter_defaults = dict(
    mask_ratio=0.15,
    epochs=10,
    max_seq_len=600,
    n_bins=51,
    CCE=False,  # Contrastive cell embedding objective
    MVC=True,  # Masked value prediction for cell embedding
    ECS=True,  # Elastic cell similarity objective
    ecs_threshold=0.85,
    lr=1e-3,
    batch_size=16,
    embsize=64,  # embedding dimension
    d_hid=64,  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
)

run = wandb.init(
    config=hyperparameter_defaults, project="scFormer", entity="scformer", reinit=True
)
config = wandb.config

# %%
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
ADV = False  # Adversarial training for batch correction
DAB = True  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer

# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_value = "auto"  # for masked values, now it should always be auto
include_zero_gene = True  # if True, include zero genes among hvgs in the training
# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
# TODO: support setting output style separately, it is easy to do using torch dataset
output_style = "binned"  # "normed_raw", "log1p", or "binned"
INPUT_BATCH_LABELS = False
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
explicit_zero_prob = True and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training
per_seq_batch_sample = True
DSBN = True and per_seq_batch_sample  # Domain-spec batchnorm
eval_batch_size = 16
schedule_interval = 1
# logging
log_interval = 100
do_eval_scib_metrics = True
lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
n_hvg = 1200
# try tuning the batch_size

# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = config.n_bins + 1
    pad_value = config.n_bins  # for padding gene expr values
    n_input_bins = config.n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = config.n_bins
if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False

# %%

save_dir = Path(
    f"./save/dev_pancreas{'_adv' if ADV else ''}{'_cce' if config.CCE else ''}"
    f"{'_cls' if CLS else ''}"
    f"{'_zeroin' if include_zero_gene else ''}"
    f"{'_mvc' if config.MVC else ''}{'_ecs' if config.ECS else ''}"
    f"{'_dab' if DAB else ''}"
    f"{'_hvg' if n_hvg > 0 else ''}"
    f"-{time.strftime('%b%d-%H-%M')}/"
)
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
# save the whole script to the dir
os.system(f"cp {__file__} {save_dir}")

logger = scf.logger
scf.utils.add_file_handler(logger, save_dir / "run.log")


# %% [markdown]
# ## Loading and preparing data
# pancreas file from https://github.com/theislab/scib-reproducibility and the scib benchmarked results can be found at https://theislab.github.io/scib-reproducibility/dataset_pancreas.html.

# %%
adata = sc.read(
    "../data/scib_datasets/human_pancreas_norm_complexBatch.h5ad", cache=True
)  # 16382 × 19093


# %%
# make the batch category column
adata.obs["str_batch"] = adata.obs["tech"].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels

adata.var["gene_name"] = adata.var.index.tolist()

# %%
# set up the preprocessor, use the args to config the workflow
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    # comment out the log1p since the data is already log1p transformed
    # log1p=True,  # 4. whether to log1p the normalized data
    # result_log1p_key="X_log1p",  # the key in adata.layers to store the log1p normalized data
    subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key="str_batch")

# %%
adata

# %% [markdown]
# ## Tokenize input

# %%
input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
num_types = len(set(celltypes_labels))
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

# train_data, valid_data = train_test_split(all_counts, test_size=0.1, shuffle=True)
(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)


# %%
# prin max, min and average num of non_zero genes
num_of_non_zero_genes = [
    np.count_nonzero(train_data[i]) for i in range(train_data.shape[0])
]
logger.info(f"max num of non_zero genes: {np.max(num_of_non_zero_genes)}")
logger.info(f"min num of non_zero genes: {np.min(num_of_non_zero_genes)}")
logger.info(f"average num of non_zero genes: {np.mean(num_of_non_zero_genes)}")
logger.info(
    f"99% quantile num of non_zero genes: {np.quantile(num_of_non_zero_genes, 0.99)}"
)
logger.info(f"max original values: {np.max(train_data)}")
logger.info(
    f"average original non_zero values: {np.mean(train_data[np.nonzero(train_data)])}"
)
logger.info(
    f"99% quantile original non_zero values: {np.quantile(train_data[np.nonzero(train_data)], 0.99)}"
)
logger.info(f"num of celltypes: {num_types}")
if not include_zero_gene:
    max_seq_len = min(
        config.max_seq_len, np.max(num_of_non_zero_genes) + 1
    )  # 1 for <cls>


# %%
# # clip the train data to the 99% quantile
# max_value = np.quantile(train_data[np.nonzero(train_data)], 0.99)
# train_data = np.clip(train_data, None, max_value)
# valid_data = np.clip(valid_data, None, max_value)
# all_counts = np.clip(all_counts, None, max_value)
max_value = np.max(train_data)


# %%
# show histogram of the clipped values
fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.hist(
    [
        train_data[np.nonzero(train_data)].flatten(),
        valid_data[np.nonzero(valid_data)].flatten(),
    ],
    bins=np.arange(0, max_value + 1, 1) + 0.5,
    label=["train", "valid"],
    density=True,
    histtype="bar",
    linewidth=2,
    rwidth=0.85,
    color=["blue", "red"],
)
ax.legend()
ax.set_xlabel("counts")
ax.set_ylabel("density")
ax.set_title("Histogram of clipped values")
plt.show()


# %%
vocab = Vocab(
    VocabPybind(genes + special_tokens, None)
)  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)


# %%
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=include_zero_gene,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)


# %%
def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=config.mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=config.mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                config.batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


# %% [markdown]
# # Create and train scFormer

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    config.embsize,
    config.nhead,
    config.d_hid,
    config.nlayers,
    nlayers_cls=3,
    n_cls=num_types,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=config.ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
).to(device)
wandb.watch(model)

if ADV:
    discriminator = AdversarialDiscriminator(
        d_model=config.embsize,
        n_cls=num_batch_types,
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


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()


criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(
        optimizer_dab, schedule_interval, gamma=0.9
    )
if ADV:
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
    scheduler_E = torch.optim.lr_scheduler.StepLR(
        optimizer_E, schedule_interval, gamma=0.9
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
    scheduler_D = torch.optim.lr_scheduler.StepLR(
        optimizer_D, schedule_interval, gamma=0.9
    )


def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    (
        total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        output_dict = model(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels if INPUT_BATCH_LABELS or DSBN else None,
            CLS=CLS,
            CCE=config.CCE,
            MVC=config.MVC,
            ECS=config.ECS,
            do_sample=do_sample_in_train,
        )

        masked_positions = input_values.eq(mask_value)  # the postions to predict
        loss = loss_mse = criterion(
            output_dict["mlm_output"], target_values, masked_positions
        )
        if explicit_zero_prob:
            loss_zero_log_prob = criterion_neg_log_bernoulli(
                output_dict["mlm_zero_probs"], target_values, masked_positions
            )
            loss = loss + loss_zero_log_prob
        if CLS:
            target_labels = torch.tensor(train_labels[i : i + config.batch_size]).to(
                device
            )
            loss_cls = criterion_cls(output_dict["cls_output"], target_labels)
            loss = loss + loss_cls
        if config.CCE:
            loss_cce = 10 * output_dict["loss_cce"]
            loss = loss + loss_cce
        if config.MVC:
            loss_mvc = criterion(
                output_dict["mvc_output"], target_values, masked_positions
            )
            loss = loss + loss_mvc
        if config.MVC and explicit_zero_prob:
            loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                output_dict["mvc_zero_probs"], target_values, masked_positions
            )
            loss = loss + loss_mvc_zero_log_prob
        if config.ECS:
            loss_ecs = 10 * output_dict["loss_ecs"]
            loss = loss + loss_ecs
        if DAB:
            # try weighting and separate optimizer
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss = loss + loss_dab

        # TODO: test w/o these and only train adversarial
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if ADV:
            # rerun the model for adversarial training
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or DSBN else None,
                CLS=CLS,
                CCE=config.CCE,
                MVC=config.MVC,
                ECS=config.ECS,
                do_sample=do_sample_in_train,
            )

            # TRAINING DISCRIMINATOR
            loss_adv_D = criterion_adv(
                discriminator(output_dict["cell_emb"].detach()), batch_labels
            )
            if epoch > adv_D_delay_epochs:
                discriminator.zero_grad()
                loss_adv_D.backward()
                optimizer_D.step()

            # TRAINING ENCODER
            loss_adv_E = -criterion_adv(
                discriminator(output_dict["cell_emb"]), batch_labels
            )
            # NOTE: the loss is negative here because we want to maximize
            # the cross_entropy_loss, in other words, disguise against the discriminator
            # TODO: test adv_E_delay_epochs
            if epoch > adv_E_delay_epochs:
                model.zero_grad()
                discriminator.zero_grad()
                loss_adv_E.backward()
                optimizer_E.step()

        # # TRAINING DISCRIMINATOR
        # if ADV:
        #     discriminator.zero_grad()
        #     loss_adv_D = criterion_adv(
        #         discriminator(output_dict["cell_emb"].detach()), batch_labels
        #     )
        #     loss_adv_D.backward()
        #     optimizer_D.step()

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_cls += loss_cls.item() if CLS else 0.0
        total_cce += loss_cce.item() if config.CCE else 0.0
        total_mvc += loss_mvc.item() if config.MVC else 0.0
        total_ecs += loss_ecs.item() if config.ECS else 0.0
        total_dab += loss_dab.item() if DAB else 0.0
        total_adv_E += loss_adv_E.item() if ADV else 0.0
        total_adv_D += loss_adv_D.item() if ADV else 0.0
        total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
        total_mvc_zero_log_prob += (
            loss_mvc_zero_log_prob.item() if config.MVC and explicit_zero_prob else 0.0
        )
        total_error += mre.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if CLS else 0.0
            cur_cce = total_cce / log_interval if config.CCE else 0.0
            cur_mvc = total_mvc / log_interval if config.MVC else 0.0
            cur_ecs = total_ecs / log_interval if config.ECS else 0.0
            cur_dab = total_dab / log_interval if DAB else 0.0
            cur_adv_E = total_adv_E / log_interval if ADV else 0.0
            cur_adv_D = total_adv_D / log_interval if ADV else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
            )
            cur_mvc_zero_log_prob = (
                total_mvc_zero_log_prob / log_interval
                if config.MVC and explicit_zero_prob
                else 0.0
            )
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"cls {cur_cls:5.2f} | " if CLS else "")
                + (f"cce {cur_cce:5.2f} |" if config.CCE else "")
                + (f"mvc {cur_mvc:5.2f} |" if config.MVC else "")
                + (f"ecs {cur_ecs:5.2f} |" if config.ECS else "")
                + (f"dab {cur_dab:5.2f} |" if DAB else "")
                + (f"adv_E {cur_adv_E:5.2f} |" if ADV else "")
                + (f"adv_D {cur_adv_D:5.2f} |" if ADV else "")
                + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                + (
                    f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                    if config.MVC and explicit_zero_prob
                    else ""
                )
            )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or DSBN else None,
                CLS=False,  # evaluation does not need CLS or CCE
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=do_sample_in_train,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = input_values.eq(mask_value)
            loss = criterion(output_values, target_values, masked_positions)
            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            total_num += len(input_gene_ids)
    return total_loss / total_num, total_error / total_num


# %%
best_val_loss = float("inf")
best_model = None


for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=eval_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )
    train(
        model,
        loader=train_loader,
    )
    val_loss, val_mre = evaluate(
        model,
        loader=valid_loader,
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
    wandb.log({"val_loss": val_loss, "val_mre": val_mre})
    if DAB_separate_optim:
        scheduler_dab.step()
    if ADV:
        scheduler_D.step()
        scheduler_E.step()


# %%
# compare with the naive baseline of all ones
predict_ones = torch.ones(valid_data_pt["gene_ids"].shape, dtype=torch.float32)
mse = masked_mse_loss(
    predict_ones, valid_data_pt["target_values"], valid_data_pt["values"].eq(mask_value)
)
mre = masked_relative_error(
    predict_ones, valid_data_pt["target_values"], valid_data_pt["values"].eq(mask_value)
)
logger.info(f"MSE: {mse.item()}, MRE: {mre.item()}")

# %%
# save the best model
torch.save(best_model.state_dict(), save_dir / "best_model.pt")


# %% [markdown]
# # Analysis

# %%
# load the best model
model = TransformerModel(
    ntokens,
    config.embsize,
    config.nhead,
    config.d_hid,
    config.nlayers,
    nlayers_cls=3,
    n_cls=num_types,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=config.ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
)
model.load_state_dict(torch.load(save_dir / "best_model.pt"))
model.to(device)
model.eval()

# %% [markdown]
# ## Gene embeddings

# %%
# the model encoder generates **normalized** embeddings
gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))
gene_embeddings = gene_embeddings.detach().cpu().numpy()


# %%
sc.tl.rank_genes_groups(adata, groupby="celltype", method="wilcoxon")
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

celltypes = adata.obs["celltype"].unique().tolist()

gene_groups = {}
for ct in celltypes:
    top6_markers = adata.uns["rank_genes_groups"]["names"][ct][:6].tolist()
    marker_genes = []
    for m in top6_markers:
        # TODO: should have all makers in adata, check out here
        if m in adata.var.index:
            marker_genes.append(adata.var.loc[m, "gene_name"])
    if len(marker_genes) > 0:
        gene_groups[ct] = marker_genes

gene_groups

# %%
# TODO: need some annotations like pathways
# pca of the gene embeddings
from sklearn.decomposition import PCA


gene_embeddings_pca = PCA(n_components=30).fit_transform(gene_embeddings)


# umap of the gene embeddings
import umap

embedding_umap = umap.UMAP().fit_transform(gene_embeddings)
# plot the umap
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
ax.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c="gray", alpha=0.5)
# highlight the gene groups
# color selet from tab20
colors = sns.color_palette("tab20", len(gene_groups))
for i, (k, v_list) in enumerate(gene_groups.items()):
    ax.scatter(
        embedding_umap[vocab(v_list), 0],
        embedding_umap[vocab(v_list), 1],
        color=colors[i],
        alpha=0.9,
        label=k,
    )
    for v in v_list:
        ax.text(
            embedding_umap[vocab[v], 0] + 0.04,
            embedding_umap[vocab[v], 1],
            v,
            fontsize=6,
            alpha=0.9,
            fontweight="bold",
        )
ax.legend(fontsize=6)
ax.set_title("UMAP of the gene embeddings")
plt.tight_layout()
plt.savefig(save_dir / "gene_embeddings_umap.png")

## Cell embeddings
if per_seq_batch_sample:
    # sort the adata by batch_id in advance
    adata = adata[adata.obs["batch_id"].argsort()].copy()
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = adata.obs["celltype"].tolist()
    num_types = len(set(celltypes_labels))
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)

# %%
# weighted using raw counts, non softmaxed
# cell_embeddings = np.matmul(np.array(all_counts), gene_embeddings)
# cell_embeddings = cell_embeddings / np.linalg.norm(
#     cell_embeddings, axis=1, keepdims=True
# )


# %%
# adata.obsm["X_scFormer"] = cell_embeddings

# # %%
# sc.pp.neighbors(adata, use_rep="X_scFormer")
# sc.tl.umap(adata, min_dist=0.3)

# %%
# sc.pl.umap(
#     adata,
#     color=["str_batch", "celltype"],
#     ncols=2,
#     frameon=False,
#     show=False,
# )
# plt.tight_layout()
# plt.savefig(save_dir / "embeddings_umap[weighted_gene].png", dpi=300)

# %% [markdown]
# **Note:** the weighting choice is really important: raw counts weighting works much better than the softmax weighting. I believe this is because softmax include the large amount of zero-valued genes in the cell embedding, which introduce noises. While raw counts weighting is more focused on the most characteristic subset of genes.
#
# Overall, it is worth exploring posible weighting choices.

# %% [markdown]
# **Note**: the batch removal looks fine even on this only 10k cells. One reason is that I still believe the composition of non-zero genes itself is the most informative feature, and it doesn't really need to be related to the learning process. The same celltype across different batches would have similar subesets of non-zero genes. Different celltypes always have different sets of non-zero reading genes. (**TODO: we can actually measure this**.) So this convey quite important biology information right coming out of the raw sequencing data.
#
# This reflects my previous assumptions why simple tf-idf would work well for single cell integrations.

# %% [markdown]
# ### clustering

# %%
# the scanpy leiden used the above computed neighbors
# sc.tl.leiden(adata, key_added="leiden_scFormer", resolution=0.5)

# # %%
# sc.pl.umap(
#     adata,
#     color=["leiden_scFormer"],
#     frameon=False,
# )

# %%
# target_labels = category_str2int(adata.obs["celltype"].values)
# pred_labels = category_str2int(adata.obs["leiden_scFormer"].values)

# # ARI and NMI
# ari = adjusted_rand_score(target_labels, pred_labels)
# nmi = normalized_mutual_info_score(target_labels, pred_labels)
# logger.info(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")


# %% [markdown]
# Non-weighting gene embeddings:
# - 10 epochs: ARI: 0.798, NMI: 0.754
# - 30 epochs: ARI: 0.814, NMI: 0.760
#
# weighted gene embeddings:
# - 30 epochs: ARI: 0.616, NMI: 0.750" if resolution=0.2, ARI: 0.837, NMI: 0.799
#
# cls embeddings:
# - ARI: 0.631, NMI: 0.638
#
# ### Update with three-fc-layer cls decoder
# Non-weighting gene embeddings:
# - 30 epochs: ARI: 0.846, NMI: 0.782
#
# weighted gene embeddings:
# - 30 epochs: ARI: 0.640, NMI: 0.773
#
# cls embeddings:
# - ARI: 0.346, NMI: 0.634

# %% [markdown]
# ### scib metrics

# %%
import scib

# Wrapper for all scib metrics, we leave out some metrics like hvg_score, cell_cyvle,
# trajectory_conservation, because we only evaluate the latent embeddings here and
# these metrics are evaluating the reconstructed gene expressions or pseudotimes.
def eval_scib_metrics():
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key="str_batch",
        label_key="celltype",
        embed="X_scFormer",
        isolated_labels_asw_=True,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=True,
        clisi_=True,
    )

    logger.info(f"{results}")

    result_dict = results[0].to_dict()
    logger.info(
        "Biological Conservation Metrics: \n"
        f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
        f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
        "Batch Effect Removal Metrics: \n"
        f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
        f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    )
    return result_dict


# %%
# if do_eval_scib_metrics:
#     results = eval_scib_metrics()

# %% [markdown]
# ## Random embeddings baseline

# %%
random_embeddings = np.random.normal(size=gene_embeddings.shape)
cell_embeddings = np.matmul(np.array(all_counts), random_embeddings)
cell_embeddings = cell_embeddings / np.linalg.norm(
    cell_embeddings, axis=1, keepdims=True
)

# %%
adata.obsm["X_scFormer"] = cell_embeddings
sc.pp.neighbors(adata, use_rep="X_scFormer")
sc.tl.umap(adata, min_dist=0.3)
sc.pl.umap(
    adata,
    color=["str_batch", "celltype"],
    ncols=2,
    frameon=False,
    show=False,
)
plt.tight_layout()
plt.savefig(save_dir / "embeddings_umap[random_gene].png", dpi=300)

# %%
if do_eval_scib_metrics:
    results = eval_scib_metrics()

# %% [markdown]
# ## Evaluate avg pooling embeddings

# %%
# avg pooling of last layer embedding
# tokenized_all = tokenize_and_pad_batch(
#     all_counts,
#     gene_ids,
#     max_len=config.max_seq_len,
#     vocab=vocab,
#     pad_token=pad_token,
#     pad_value=pad_value,
#     append_cls=True,  # append <cls> token at the beginning
#     include_zero_gene=include_zero_gene,
# )
# all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
# src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
# with torch.no_grad():
#     ouput_embeddings = model.encode_batch(
#         all_gene_ids,
#         all_values.float(),
#         src_key_padding_mask=src_key_padding_mask,
#         batch_size=eval_batch_size,
#         batch_labels=torch.from_numpy(batch_ids).long()
#         if INPUT_BATCH_LABELS or DSBN
#         else None,
#     )[:, 1:, :]
#     ouput_embeddings = ouput_embeddings.detach().cpu().numpy()

# cell_embeddings = np.mean(ouput_embeddings, axis=1)

# %%
# adata.obsm["X_scFormer"] = cell_embeddings
# sc.pp.neighbors(adata, use_rep="X_scFormer")
# sc.tl.umap(adata, min_dist=0.3)
# sc.pl.umap(
#     adata,
#     color=["str_batch", "celltype"],
#     ncols=2,
#     frameon=False,
#     show=False,
# )
# plt.tight_layout()
# plt.savefig(save_dir / "embeddings_umap[avg_pooling].png", dpi=300)

# # %%
# if do_eval_scib_metrics:
#     results = eval_scib_metrics()

# %% [markdown]
# ## Evaluate weighted pooling embeddings

# %%
# cell_embeddings = np.sum(ouput_embeddings * all_values.numpy()[:, 1:, None], axis=1)
# cell_embeddings = cell_embeddings / np.linalg.norm(
#     cell_embeddings, axis=1, keepdims=True
# )

# # %%
# adata.obsm["X_scFormer"] = cell_embeddings
# sc.pp.neighbors(adata, use_rep="X_scFormer")
# sc.tl.umap(adata, min_dist=0.3)
# sc.pl.umap(
#     adata,
#     color=["str_batch", "celltype"],
#     ncols=2,
#     frameon=False,
#     show=False,
# )
# plt.tight_layout()
# plt.savefig(save_dir / "embeddings_umap[weighted_pooling].png", dpi=300)

# %%
# if do_eval_scib_metrics:
#     results = eval_scib_metrics()

# %% [markdown]
# ## Evaluate cls cell embeddings

# %%
# cls embedding
tokenized_all = tokenize_and_pad_batch(
    all_counts,
    gene_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=include_zero_gene,
)
all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
with torch.no_grad():
    cell_embeddings = model.encode_batch(
        all_gene_ids,
        all_values.float(),
        src_key_padding_mask=src_key_padding_mask,
        batch_size=eval_batch_size,
        batch_labels=torch.from_numpy(batch_ids).long()
        if INPUT_BATCH_LABELS or DSBN
        else None,
    )[:, 0, :]
    cell_embeddings = cell_embeddings.detach().cpu().numpy()
cell_embeddings = cell_embeddings / np.linalg.norm(
    cell_embeddings, axis=1, keepdims=True
)

# %%
adata.obsm["X_scFormer"] = cell_embeddings
sc.pp.neighbors(adata, use_rep="X_scFormer")
sc.tl.umap(adata, min_dist=0.3)
sc.pl.umap(
    adata,
    color=["str_batch", "celltype"],
    ncols=2,
    frameon=False,
    show=False,
)
plt.tight_layout()
plt.savefig(save_dir / "embeddings_umap[cls].png", dpi=300)

# %%
if do_eval_scib_metrics:
    results = eval_scib_metrics()

# log image
# log model
avg_bio = (
    results["NMI_cluster/label"] + results["ARI_cluster/label"] + results["ASW_label"]
) / 3
wandb.log(
    {
        "example": wandb.Image(
            os.path.join(save_dir, "embeddings_umap[cls].png"),
            caption="CLS output avg_bio={}".format(avg_bio),
        )
    }
)

wandb.log(
    {
        "example1": wandb.Image(
            os.path.join(save_dir, "gene_embeddings_umap.png"),
            caption="CLS output avg_bio={}".format(avg_bio),
        )
    }
)

wandb.log(results)
wandb.log({"avg_bio": avg_bio})

artifact = wandb.Artifact("model_avg_bio_{}".format(avg_bio), type="model")
glob_str = os.path.join(save_dir, "best_model.pt")
artifact.add_file(glob_str)
run.log_artifact(artifact)

wandb.finish()
gc.collect()
