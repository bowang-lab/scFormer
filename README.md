# scFormer

## Installation

For developing

```bash
$ git clone this-repo-url
$ cd scFormer
$ poetry install
```

## Known Issues

1. For pytorch<=1.10 use `pip install setuptools==59.5.0` for a dependency
   [issue](https://github.com/pytorch/pytorch/pull/69904) with setuptools.
2. To install a torch version compatable with specific cuda version or device requirement, like A100. We recomend after the abover installation steps, mannually reinstall the needed torch version via pip lised [here](https://pytorch.org/get-started/previous-versions/).

## Paper Plan

1. [optional] The transformer method paper. [Here](https://docs.google.com/presentation/d/10nGBgwQ7hKvprjDjmfk7wBhbmbkvkAj6WkMo5hcATgQ)
2. The transformer pretraining paper. [Here](https://docs.google.com/presentation/d/10nGBgwQ7hKvprjDjmfk7wBhbmbkvkAj6WkMo5hcATgQ)
3. Understanding single cell pretraining paper.

### Roadmap toward Milestone 1

See also at the [scFormer Project](https://github.com/users/subercui/projects/1) board

1. Single dataset (get satisfying results- 9.10)
2. Integration task （get satisfying results- 9.10）
   1. [x] Preprocessing, normalizing (8.26 - 8.29)
   2. [x] Adversarial training for batch effect removal (8.29 - 9.1)
   3. Adding pathway embeddings, the implementation part (9.1-9.3)
   4. Hyperparameter search and experiments to improve performance (9.3-9.10)
3. Perturbation task （get satisfying results -9.10）
   1. Zeros, preprocessing, normalizing, distributional prediction (8.26 - 8.29)
   2. Update training from random projection to DeepVelo style (8.29-9.1)
   3. Importance weighting on differential genes in training (9.1-9.3)
   4. Adding pathway embeddings (9.1-9.3)
   5. Hyperparameter search and experiments to improve performance (9.3-9.10)
4. GRN task ( 9.10 - 9.15)
5. Manuscript (9.10 - 9.28)
   1. Outline ready before 9.10
6. Ablation studies and other analysis studies (9.10 -9.28)
7. Other optional tasks (9.15 - 9.28)
   1. Integrate huggingface
   
Haotian:
- Search for integration task (9.22, 9.23)
- Fill in the section for integration task (9.24)

Chloe:
- Experiment for single dataset task (9.22, 9.23)
- Search for single dataset task (9.23, 9.24)
- Fill in the section for single dataset task (9.25)

Haotian:
- Add updates for perturbation task (9.22)
- Search for perturbation task (9.23, 9.24)
- Fill in the section for perturbation task (9.25)

- Experiment for GRN task (9.26)
- Fill in the section for GRN task (9.27)

Chloe:
- Ablation study experiments (9.26)
- Fill in the section for ablation study (9.27)

Both:
- Final manuscript updates and refine (9.28)

## The Data Structure for Large-Scale Computing

Building up the data structure with large-scale computing bear in mind, support accessing and grouping cells across studies:

### Key Features for 10+ Million Data

- fast full splicing and indexing across studies
- data streaming
- easily appending new data or removing studies, without constraint of the gene dimensions
- runtime data object of hybrid memory and disk storage
- tracking, synchronizing and versioning of data changes
- maximizing interpretability if saving in json, the on disk directory and files are self explanatory to a large extent
- efficient compression and loading if saving in parquet

### Data Schema

1.  The key structure of scBank is the datatables. Each datatable essentially contains rows of data, each row per cell. Firstly, there will alway be a **main** datatable, which has no difference to other datatables, only its name will be indicated by the `main_data` field in the `manifest.json` file.

    `example_main.datatable.jsonl`:

    ```jsonc
    {
      "id": "cell_id", // required
      "genes": [gene_id_1, gene_id_3, ...],  // used if data is sparse
      "expressions": [value_1, value_3, ...],
    },
    ...

    ```

    We support additonal cell-specific contents like nromalized expressions, etc. Each additional data will be stored in an separate datatable.

    An example of data in consecutive keys and values, usually can be used to store sparse cell-gene expressions,

    `normalized_expression.datatable.jsonl`:

    ```jsonc
    {
      "id": "cell_id",  // required
      "genes": [gene_id_1, gene_id_3, ...],  // used if data is sparse
      "expressions": [value_1, value_3, ...],
    },
    ...

    ```

    An example of data containing only dense values. Using this assumes cells having the same number of dimensions/columns, for example, like the umap coordinates, latent embedding, etc. The dim/col name can be specified in the study table,

    `some_dense_data.datatable.jsonl`:

    ```jsonc
    {
      "id": "cell_id",  // required
      "row_name": [value_1, value_2, ...], // find column keys in study table
    },
    ...

    ```

    **Note**: the difference between the two types of datatable is the number of fields. scBank will use this to load and maintain the data correctly, so the top level fields should always be id, [custom key name], custom value name.

2.  The cell metatable to store cell-specific information, such as cell type, etc.

    `cellmeta.jsonl`:

    ```jsonc
    {
      "id": "cell_id",  // required
      "meta": {  // required
        "study": "study_id",  // required
        "cell_type": "cell_type",
        "cell_line": "cell_line",
        "disease": "disease",
        "tissue": "tissue",
        "age": "age",
        },
    },
    ...

    ```

3.  The study table is like a group of study cards. Each study card has information like study metadata, the cell ids that belong to the study. Study metadata include copy numbers, hvgs, cell type set, etc.

    `studytable.jsonl`:

    ```jsonc
    {  // a study card
      "id": "study_id",  // required
      "cells": [  // required
        "cell_id_1",
        "cell_id_2",
        ...
        ],
      "meta": {
        "cell_types": ["cell_type_1", "cell_type_2", ...],
        "hvgs": ["gene_id_1", "gene_id_3", ...],
        "copy_number": {
          "gene_id_1": copy_number_value_1,
          "gene_id_3": copy_number_value_3,
          ...
          },
        },
      "key_map": {  // optional, the column keys for dense datatables
        "some_dense_data": [gene_id_1, gene_id_2, ...],
        ...
        },
    },
    ...

    ```

4.  The paired gene vocabulary to link the gene_id to gene_name. **Note**: we can also have a celltype vocabulary to make sure the celltypes are represented in shared ids across studies.

    `gene.vocab.json`:

    ```jsonc
    {
      "1": "gene_name_1",  // required
      "2": "gene_name_2",
      ...
    }
    ```

5.  Gene annotation table. In theory, some gene annotations do not need to be associated with the studies.

    `gene.annotation.jsonl`:

    ```jsonc
    {
      "id": "gene_id", // required
      "function": "function_1",
      "total_variance": total_variance_1,
      "alias": ["alias_1", "alias_2", ...],
      ...
    },
    ...

    ```

6.  md5 checksum of the data, particularly for the gene vocabulary.

    `manifest.json`:

    ```jsonc
    {
      "gene_vocab_file_name": "md5_checksum_of_gene_vocab",  // required
      "main_data": "example_main",  // required, the name of the main datatable
      ...
    }
    ```

Overall, the data can be stored in jsonl format. Or you can really setup a mongoDB database. All 6+ files stored in a specific directory, and file metadata stored in the md5 manifest file. **Note**: the data directory should be condidered as the protected data structure of scBank. Sould only use the scBank API to access and edit the data files.

Compared to Anndata, the `X` goes to the main table content, `obs` (like celltype, tissue ...) goes to the cell metadata, global `var` (like gene_name, function, all variance ...) goes to the gene annotation table, study-specific `var` (like copy numbers ...) goes to the study table metadata, `uns` (like cell_types, hvgs ...) goes to the study table metadata, `obsm` (like umap, pca ...) goes to the additional data tables, global `varm` goes to the gene annotation table, `layers` (like normalized expressions ...) goes to the additional data tables. `obsp` and `varp` will need future support, since need to nicely support custom annotation dimensions beyond cells or genes.

## **IMPORTANT** Contributing Rules

1. Correlation and distribution plot of gene expressions as a standard evaluation for all tasks, so including not only perturbation, also integration, gene network, etc.
2. Experiment with different random seeds shoud be prioritized for all tasks and methods as one of the first jobs to do, since it do seems the performance could vary much in exsting runs.

## Jobs Info

Pretrain jobs:

<!-- prettier-ignore -->
| Job Link | Save Directory | MSE |
| -------- | -------------- | --- |
| https://ml.azure.com/runs/scFormer-40fd5876-fine-duckling-cellxgene_all-m1024-b32-l12-s8-e7-a4aefeca | cellxgene_all-Jul16-01-26-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-curious-foal-cellxgene_all-m1024-b32-l12-s8-b172b90c | cellxgene_all-Jul15-05-22-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-top-swift-cellxgene_all-m1024-b32-l12-s8-a9e6ca62 | cellxgene_all-Jul10-00-09-2022 | 0.7974 |
| https://ml.azure.com/runs/scFormer-40fd5876-faithful-crappie-cellxgene_all-m1024-b32-l12-s8-5e3e6077 | cellxgene_all-Jul09-19-10-2022 | 0.8101 |
| https://ml.azure.com/runs/scFormer-40fd5876-possible-humpback-cellxgene_all-m512-b64-l12-s8-9b500668 | cellxgene_all-Jun30-21-19-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-decent-albacore-cellxgene_all-m1200-b32-l12-s8-eb1d971e  | cellxgene_all-Jun30-05-55-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-destined-drum-cellxgene_all-m512-b32-l12-s8-8e607e28     | cellxgene_all-Jun30-20-35-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-native-primate-cellxgene_all-m1024-b32-l12-s8-7f9f00c7   | cellxgene_all-Jun30-05-47-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-diverse-javelin-cellxgene_all-m1024-b64-s2-4b1ee6eb      | cellxgene_all-Jun30-05-26-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-verified-dolphin-cellxgene_all-m1024-b32-s1-09fa742e     | cellxgene_all-Jun30-05-08-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-capable-mammal-cellxgene_all-m512-b64-l8-s2-bf87f7d1     | cellxgene_all-Jun29-23-54-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-exciting-cougar-cellxgene_all-m1024-b32-l8-s2-289f7a1a   | cellxgene_all-Jun29-23-52-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-viable-weasel-cellxgene_all-m1024-b64-l4-s4-1afb28b5     | cellxgene_all-Jun29-20-39-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-careful-bird-cellxgene_all-m1024-b64-s3-35303659         | cellxgene_all-Jun29-15-53-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-upright-collie-cellxgene_all-m1024-b64-l8-s2-ab8c8599    | cellxgene_all-Jun29-15-56-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-driven-civet-cellxgene_all-m1024-b64-s3-f2c1406c         | cellxgene_all-Jun29-15-45-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-main-cellxgene_f10-b64-s1-9b42ceca                       | cellxgene_f10-Jun13-18-57-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-diverse-drake-cellxgene_f10-53ec3928                     | cellxgene_f10-Jun13-18-16-2022 |
| https://ml.azure.com/runs/scFormer-40fd5876-dynamic-quetzal-cellxgene_f10-a3c3f2df                   | cellxgene_f10-Jun13-15-03-2022 |

Finetune jobs:

<!-- prettier-ignore -->
| Job Link | Save Directory | Best Epoch | MSE, MRE |
| -------- | -------------- | ---------- | -------- |
| https://ml.azure.com/runs/scFormer-40fd5876-climbing-buffalo-Finetune-pbmc-m1024-b32-2dab67c7 | finetune-pbmc-Jul18-00-05-2022 | 10 | 1.8522, 0.2725|
| https://ml.azure.com/runs/scFormer-40fd5876-loving-collie-Finetune-pbmc-m1024-b64-85aee7ca | finetune-pbmc-Jul04-02-05-2022 | 10 | 1.9981, 0.2732 |
| https://ml.azure.com/runs/scFormer-40fd5876-positive-halibut-Finetune-pbmc-m1024-b64-7204b15d | finetune-pbmc-Jul04-02-05-2022 | 6 | 2.0232, 0.2834 |
| https://ml.azure.com/runs/scFormer-40fd5876-smart-hippo-Finetune-pbmc-m1024-b64-c202b8e2 | finetune-pbmc-Jul04-01-48-2022 | 10 | 2.0044, 0.2695|

## Change Log

### Sept 18, 2022

- [ ] **IMPORTANT** Try effiecient attention methods like the [Hydra Attention](https://arxiv.org/abs/2209.07484) to directly modeling long input sequence. See issue #13
- [ ] Try ranomly mask out gene tokens that are of high expression values as and let the model predict them as well, asking the question "there is a highly expressed gene which one it should be".
- [ ] Knn classification evaluation of the embedding
- [ ] Actually can provide a handle for tuning the looking of the embeddings, like reported [here](https://www.biorxiv.org/content/10.1101/2022.07.14.500036)
- [ ] Visualizing the gene relation using attention scores.

### Sept 5, 2022

- [x] Try including zeros and also combining other techniques.
- [x] Add a point gamma rate for predicting zero expr values, such as the scvi style.
- Other techniques:
  - Does softmax over the expr reconstruction layer help for batch correction?
- Improve CCE within seq batches.

### Sept 2, 2022

- [ ] Check the evaluation code for embedding visualization, maybe there is the key. Just remind to make sure all the evaluation code is correct.
- [ ] **IMPORTANT**. Using batch token in input, so the model can easily learn some anti-batch features to remove the information. You can definitely use the input to make it much more easier for the model, no need to just forcing to learn from the objective backpropagation.
- [ ] Generally speaking for the batch correction integration task, the goal is to see the celltype clearly and meanwhile merge the batches. Now we already have the input batch infomation and methods to deal with the batch mixing, the thing we need is particularly a way to clearly point out where is the celltype. For example, we do know if already **look into the same batch, the most closely expressed cells are likely to be the same celltype. We can use this heuristics to train**

### Sept 1, 2022

- [ ] Support setting output style separately, it is easy to do using torch dataset.
- [ ] Fix the pad value issue. Check how pad value is handled in random masking, it must support pad value other than zeros. Issue #4.
- [ ] Directions to explore for batch correction #5.

- [ ] For adversarial training, replace all relu with leaky relu and try to use soft labels. Tips from [ganhacks](https://github.com/soumith/ganhacks) No. 5 and 6
- [ ] For integration task, try using more input zero genes. That might be required for batch corrections, since different seq batches may have specific bias on certain gene subsets, including more genes relieves the bias.

### Aug 30, 2022

- [ ] Pred output zeros separately using the cell emb vector. Match the zero processing between input and prediction.
- [ ] For nomalization, provide options for different type of input and output formats. Also, provide option of categorical value embeddings if the input is binned, and option of direct scaling if the input is real value.
- [ ] Control zero ratio

### Aug 23, 2022

- [ ] Let's sure try the [SCENIC+](https://www.biorxiv.org/content/10.1101/2022.08.19.504505v1) and [SCENIC](https://www.nature.com/articles/nmeth.4463) data for Gene Regualtory Netork task, the [scPerturb](https://www.scperturb.org/datasets) data and the whole [zebra fish data](https://www.biorxiv.org/content/10.1101/2022.08.04.502764v1) for the perturbation task at our proper stage.

### Aug 20, 2022

- [ ] Issue: the obvious overfitting when training perturbation adamson data

  > | scFormer - INFO - | epoch 1 | 1600/1698 batches | lr 0.0010 | ms/batch 260.64 | loss 0.10 | mse 0.10 | mre 57824.35 |
  > | scFormer - INFO - | end of epoch 1 | time: 449.15s | valid loss/mse 0.1892 | mre 100970.9667 |

- [ ] How to how do you deal with the zeros in target? Should you ignore them in the objective computation, or at least use a smaller weight? Otherwise, the model will be encouraged to predict small values in general.
  - In self-supervised settings. The target zeros may have more sense to
    learn to. Since zeros likely mean small gene expression.
  - In at learst the perturbation setting, it is not clear the zeros make
    any sense. Since that's input and target cell pairs, and at least in
    the input, the specific gene is well expressed. It sometimes can be
    just not very well paired and the target cell behaves differently.

### Aug 18, 2022

- [ ] Examine and fix the issue that eval batch size influence the preturbation prediction result values.

  The issue comes from that the non-zero ids are computed in the union of the whole batch in the `model.pred_perturb`, https://github.com/subercui/scFormer/blob/71683fa4c4a91270d24ffd641c62ca1a3cb6dbe8/scformer/model/generation_model.py#L250, therefore the non-zero postions changes when the batch size changes.

  This also actually causes inconsistency between training and testing, where in training non-zero positions are computed in each individual row. And the fact that clearly we see chaning the batch size do influence the results in inference confirms the inconsistency do matter and we should have a better way to update the workflow.

  To have a quick worthwhile test, we can first try using the batch-wise setting in training as well. To completely fix this, this comes to the discussion and our plan of the new zero-aware training process.

### Aug 16, 2022

- About the normalization, input and output:

  1. The ultimate way should be binning the input
     - separate zero representation, in other words, zero should always be one single bin.
     - use fixed sum, so the sequencing depth and other things won't matter. Eventually, the the relative abundance and relation should be able to represent enough biology.
     - use binning into always k bins, for example k can be 50, so that whether log1p or not won't make a difference.
  2. Predict distribution as in quantile instead of single value predictions.
     - In training, instead of simple categorical objective, using soft label as not one-hot target, but several weighted hots, so that the model can learn when the prediction is close. Related methods include soft ordinal classification like [this](https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html), softmax with temperature, and even as well distributed RL approaches, like [c51](https://arxiv.org/abs/1707.06887), [iqn](https://arxiv.org/abs/1806.06923), and see the introductions [here](https://flyyufelix.github.io/2017/10/24/distributional-bellman.html) and [here](https://medium.com/analytics-vidhya/distributional-reinforcement-learning-part-2-iqn-and-fqf-567fbc7a04d7)
     - weighted prediction, using the max logits and the two logits beside it. So the predicted value would be the weighted sum of the three consecutive bins.
  3. There is still space to try whther have a separate zero out probability prediction is worthwhile.
  4. Advantages of this way comparing other common choices:

     - Compared to simple real value prediction and MSE training. Even though MSE does weights on larger absolute differences, the prediction can still be oversmoothed especially when most ragets are zeros and ones.
     - Compared to predicting parameters for like negative binomial or other distribution, and then sample the value. This way has less requirements for the data distribution type, should be more suitable for training across studies and modalities.

- [ ] For the value embedding, try using log1p value or raw value times the gene embedding, and note to multiply them after the layer norm.

### Aug 11, 2022

- [ ] Consider normalization strategy as from the https://www.nature.com/articles/s41588-021-00873-4. Are there really any zero values can not be explained by the caturing rate of the sequencing?

- Choices for dealing with normalizations and the zero values:
  - binning input and output into categories
  - fixed sum normalization
  - using hvgs as input
  - ZINB or negative processing and output like in existing methods. For the preprocessing, inferring the parameters of ZINB and input the expected mean for each gene instead of the actual read.
  - poison parametrised output

### Aug 8, 2022

- [x] GPU memory usage issue in the perturbation workflow. Just using clean cache to see the actual usage shows it runs fine.

- [ ] To deal with the zeros in input and potentially how to predict zero values in the output for genes.

      - Overall, should consider how the model can easily generalize between the training setting and test settings of novel cells. In the training setting, you can have whatever you want. In the embedding task test setting, you still can have whatever you want, but you should consider the scenario where even for the same celltype, different set of genes are in readout. **So the model should be able to learn something to handle this flexibility by learning something from the possible zeros.** In the generation task test setting, you eventually want to predict possible expression values for the whole genome, so definitely **should be able to predict those zero-read genes in the input with some meaningful values.**
      - First, check how other models deal with it, like those ZINB models.
      - Additonally, one possible option can be still in the input only use non-zero genes, in the output use the additional MVC objective, so use the cell embedding vector to predict all genome genes, or at least select a portion of additiona genes not in the input but from the whole genmome to predict where you can also compute the zero probability of these additonal genes of their original expression values, like for example 60%.

### Aug 7, 2022

- The computation in the GEARS paper, using global non-zero perturbgenes is a bit quistionable/problematic? Firstly, using the same setting and compare, then use a more proper/realistic setting for all and compare.

### Aug 4, 2022

- [ ] Datasets for perturbation task https://github.com/theislab/sc-pert.

- [ ] Tasks for perturbation from the sc-pert review. We should support all of them, particularly at least **A. response prediction B. Target prediction and C. interaction prediction**.
      <img src="https://ars.els-cdn.com/content/image/1-s2.0-S2405471221002027-gr1.jpg" width="60%">

### July 31, 2022

- [ ] How to make the objectives for the perturbation task? Three options, the first one makes the most sense:

  - [ ] Use DeepVelo style learning, find the possible perturbed target cells in the "perturbed neighbohood". The main model output should be the positional gene expressions, can have the matching of cell embeddings as accessory objective.
  - [ ] scGen style learning, but instead of simply computing linear delta. Learn a transformation mapping bettween from the control to the perturbed distribution. **Note**, this means instead of predicting expression values, should predict the distribution of gene expressions. Or at least work on the distribution of cell embeddings, from K original cell embeddings compute a distribution then project to the target distribution to match the target K' learned perturbed cell embeddings. So the transformation can be a normalizing flow or something. In a word, this option learn the transformation/normalizing flow between the cell embedding distributions and/or the gene expression distributions.
  - [ ] The Gears style learning, use random projections to the same celltype but perturbed cell group.
  - [ ] scGen style learning - learn cell embeddings and compute the difference delta.

  **Note**, all previously mentioned options can be combined with complementary objectives, like classify the ctrl/perturb from cell embeddings, contrastive learning objectives, etc.

- [ ] Issue for the perturbation task, how to better handle the generation of new non-zero genes that only appear in the perturbed cells?

### July 27, 2022

- [ ] For the gene expression value input and output, use distribution estimate instead of point estimate. Two options:
  - [ ] Parse value of each study into relative bins (like quantile 5%, 10%, ... 100%), then use softmax prediction. **The advantage of this option is that it make sure the encoding makes sense across studies**, because it is essentially for relative values.
  - [ ] Use poisson or binomial modeling, predict the lambda parameter.
- [ ] Add the task of design the optimal perturbation, which is like the encoding input. See the [PerturbNet](https://doi.org/10.1101/2022.07.20.500854) paper. More particularly, see the [Gears](https://www.biorxiv.org/content/10.1101/2022.07.12.499735) paper, which is highly related to what we are doing in the sense that it forms the task as from gene perturbations to predicting resulting gene expressions. Eventually, show the advantages/contributions for this perturbation task, including,
  - [ ] The ability to predict new biologically meaningful phenotypes, like the analysis in Gears.
  - [ ] non-additive effects of combinatorial perturbation and identifies genetic interaction subtypes, like the analysis in Gears.
  - [ ] search combinatorial perturbation space for novel genetic interactions, like the analysis in Gears.
  - [ ] Emphasize this is an important contribution to train on mixed celltypes and conditions, and also do pretraining and transfer learning. This greatly exceeds the ability of the other previous methods like Gears.

### July 20, 2022

- [x] Try replacing the cls_emb with avg_pool or w_pool in the cell embedding related objectives, and use them as cell embeddings.

- [ ] Also, it might be wise to increase the MLM probability. I feel the data is self is quite strong related. So in this setting, may need a more challenging task to guide the model to learn the representation. So increase the MLM probability, and see how MAE works in their settings since this sounds like a similar challenge.
- [ ] A related question is to guatantee that the model leearns to predict expression values from the the relation to other gene expression, not from the those gene names. This is particularly important, in the current arch, it is possible the model via backprop can learn average expression values for each gene, and then just predict that. Although this is is exactly what NLP transformer did, but notice that we even have the query token available here in the model. So, **Even one might argue learn average value from other gene tokens is feasible, and maybe fundamental for model when expression values are not reliable and data size not large enough. However, obviously learn the average value from the query gene token itself is not acceptable, obviously a short cut, doesn't represent any gene expression relations. So we should at least make sure query token is presented to the output layer in a not direct way, or detached.** (On the other hand, since one altimately has to ask the model what is the value of a specific gene, and then the model sure can get some prior info from the gene identity. Oh, that gene, it is usually highly expressed in these cells. In other words, one can still argue learning from the query token is natural. The key question/right question should instead be it is OK there is gene token information, you should simply use larger data so that there is diversed expression for individual gene, and then you should consider how to have model archs to promote learning from other gene relations.)

  - This can be relieved by maybe encoder-decoder arch like MAE.
  - And/or by adding augmented data with imputation, mask gene names, or switch gene names?

  **Note:**
  No matter what archs to use, several rules are all applicable. 1. try not to have the query gene embedding closely to the output value prediction layer, and also try not to have the output value prediction backprop to query gene embedding. Both are because that, the model may easily learn some default average value by backpropagation to the gene embedding.

### July 19, 2022

- [ ] Compare the MSE and MRE of different finetuned, pretrained, and from-scratch models. Make sure they are good indicators.

### July 18, 2022

- [ ] Try set the input value for the special and metainfo tokens, including `cls`, `pad`, `sep`, etc, with a specitial value, other than 0.
- [ ] Using celltype supervision, also find out the result on slide page 53.
- [ ] **About the challenge of making cell embedding more separable** into biological meaningfull clusters:

  I think one of the reasons making it not so separable now is that this is no direct strong supervision/objective to predict biological information directly from the cell embedding, while the gene embeddings are currently doint the biological prediction job in the MLM, cell embeddings are more or less just sum/average of the gene embeddings (which natually also smooth things out).

  So to solve this problem, we need **cell embeddings do the biological tasks too! This should be the key point her**. This should be even more important than the CCE task. For example, let's add a task to **directly predict MLM gene values directly from the cls cell embeddings** too. And hopefully, this will separate the cell embeddings well.

  Also, generally this is strongly related to learn a more separable sentence embedding. So have a review at that area and see if any approach can be used.

### July 17, 2022

- [ ] Try pretraining with hvgs on different datasets or for genes expressed larger than a certain thres, maybe that's where the core information comes from for each cell, and the other less expressed genes may introduce more noise.

- [ ] So the connection between query data and the pretrained/reference data can also be important. So go to find the pairs of query and reference datasets from the atlas referencing papers. These dataset connections should be more well established.

### July 16, 2022

- [ ] Add evaluation tasks, perturbation like scGen, annotation tasks like from scHPL and others.
- Another contribution of the transformer pretraining is that it offers a more robust and automatic way to deal with batch effects by augmentation just in the data space.

### July 15, 2022

- [ ] Check the gradients and the bumping of the computed mre during training, see if any issue.

### July 12, 2022

- [ ] Add the idea of distributional normalizing in the preprocessing, and prior in the output space, such as the one in the [deep count autoencoder](https://www.nature.com/articles/s41467-018-07931-2.), also the trVAE.
- Potential contribution on the "interpretability" side, to really show and examine the driving force, like purtubation settings and the gene regulation. One important benifit is that as long as we start viewing each cell as "a sentence of tens of thousands of genes", we actually go beyond the view of integrating cells or some cell embeddings and are more importantly building up the very language of biology in its original space, so that we can answer a lot of **predictions** and sort of **causal questions**, and we can **generalize to new cells** quite well (because intrinsically they are built up from the same biological language).

### July 10, 2022

- [x] Check the finetuning learning rate with warm-up ratio.
- [ ] Add baselines, pca, random gene embeddings, all zero predictions, etc.

### July 9, 2022

- [ ] **NOTE** When comparing methods, make sure you used the same preprocessing workflow, for example, select the same set of hvgs.
- Checkout the new HCA loading tool, [Galaxy](https://twitter.com/ExpressionAtlas/status/1151797848469626881)
- Checkout the using of lung atlas in large scale training, from the paper [here](https://twitter.com/mohlotf/status/1545328933247373312)

### June 30, 2022

- [ ] Consider using some batch covariant removal feature/objective in the encoder or decoder. Encourage batch-effect removing by adding the NB likelihood objective.
- [x] Using just one process to write file whenever possible in DDP training, this should solve the issue #2.

### June 29, 2022

- [x] Solve the problem of one GPU taking so much memory, for example, in this [job link](https://ml.azure.com/runs/scFormer-40fd5876-holy-burro-cellxgene_f10-m1024-b64-l8-s2-eee26019) GPU 15 used 90% memory while the other only used ~30% memory.

  This can be solved by setting the CUDA_VISIBLE_DEVICES to only one specific GPU for each process, i.e. `os.environ["CUDA_VISIBLE_DEVICES"] = str(args.local_rank)`.
  See the following discussions.

  - https://discuss.pytorch.org/t/is-it-expected-for-distributeddataparallel-to-use-more-memory-on-1-gpu-in-a-1gpu-1process-setup/77748/8
  - https://discuss.pytorch.org/t/ddp-taking-up-too-much-memory-on-rank-0/62443/4
  - https://discuss.pytorch.org/t/distributeddataparallel-got-more-than-one-processes-in-one-gpu/94382/3

### June 28, 2022

- [x] Make CCE work with data parallel training. Gathering all batch embeddings across GPU devices when computing the contrastive objective.

### June 25, 2022

- Now the default padding value for expression counts is 0, consider whether this is still appropriate if a future version uses zero-value expressed gene in input.
- About tokenizing and batch split examples using Dataset, https://huggingface.co/docs/datasets/v2.3.2/en/process#batch-processing
- [ ] We probably can just use the Dataset since it can [shuffle](https://huggingface.co/docs/datasets/v2.3.2/en/process#shuffle), [map](https://huggingface.co/docs/datasets/v2.3.2/en/process#batch-processing) and [format-transform](https://huggingface.co/docs/datasets/v2.3.2/en/process#format-transform) on the fly. Compare which is faster, loading using pytorch or this.
- [ ] Host the data as a private dataset on huggingface hub.
- [ ] For tracking the changes of datatables in scBank, can use the Dataset fingerprint and check whether it has changed.
- [ ] Use [features](https://huggingface.co/docs/datasets/v2.3.2/en/about_dataset_features) to clearly structure the column types when constructing and loading datasets in scBank.

### June 24, 2022

- [ ] Two ways to speed up the Dataset splicing, 1. use the cache feature https://huggingface.co/docs/datasets/v2.3.2/en/cache#improve-performance, 2. check the [flatten](https://huggingface.co/docs/datasets/v2.3.2/en/package_reference/main_classes#datasets.Dataset.flatten) and [flatten_indices](https://huggingface.co/docs/datasets/v2.3.2/en/package_reference/main_classes#datasets.Dataset.flatten_indices) features

### June 22, 2022

- [ ] The GeneVocab used the torchtext vocab now, which is so no flexible (e.g. it requires consecutive indices). Change the backend to a proper one.
- [ ] Update the save and load from disk for scBank.
- [ ] Can update the `__repr__` of DataBank to better show fields and show properties.

### June 21, 2022

- [ ] Try adding Curated Cancer Cell Atlas as additional interesting data and task
- [ ] The gene vocabulary should be added (1) when init, read from file. (2) Other time, add manually from object or file.
- [ ] Consider how to have auto save (when to auto save? key attributes changed, or more specifically their hash changed?), manual safe save (save all or selected attributes), sync (track flags of which self.`attr` chaged and save them), this kind of features.
- [ ] So basically, the key components to monitor are datatables and gene vocabulary. Any time datatable/vocab id added or changed (no matter in init or other places), should run the `_validate_data` and sync (Or at least warn to sync and made a buffer for them) the data between memory and disk.
- [ ] Is there a feature in python to check the delete of an object from memory? If so, we can use it to make a delayed sync feature, meaning first log changes and have changes in a buffer, then sync them to disk when User is going to close the DataBank object.
  - Basically, message the user and call the sync function in DataBank.close().
  - Also consider using a `Monitor` helper class
- [ ] Reorder the method definitions in the DataBank class.

### June 20, 2022

- [ ] Add gene vocabulary in the DataBank initialization, and add vocabulary md5 checksum in the `_validate_data`.
- [ ] Find a way to automatically call the `_validate_data` function when any data is changed.
- [ ] Move GeneVocab from scFormer to scBank.

### June 18, 2022

- [ ] 1. providing accelerated tokenization using numba/sparse/multiprocessing/dask 2. convert to instance methods 3. use it
- [ ] We might need to change the default vocab settings in DataBank, adding special tokens?
- [x] Using paquet in python, save list or np.array?
      The type in the Dataset class will always be list. So we we can preprocess in numpy, it will always be converted or saved to list.

### June 16, 2022

- [ ] Add vocabulary in the DataBank field, and sanity check during initialization.

### June 15, 2022

- [x] Do the masking for mlm in the data collator, and could have something similar like the diversified masking in huggingface

  - https://huggingface.co/docs/transformers/tasks/language_modeling
  - https://github.com/huggingface/transformers/blob/v4.19.4/src/transformers/data/data_collator.py#L748, and use the collator in data loading https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/trainer.py#L808
  - https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling

- [ ] In the scBank in disk data structure, consider make the main data and additonal data just in the same format and stored in data table files, and then let a field in the manifest json points to the main table identifier/name to indicate which one is the main data. In this way, we can easily switch and sync different main data when working with the DataBank class in runtime, by just changing the indicator field in the manifest json.

### June 14, 2022

- [ ] Could try the GraphGPS for large number of input genes.

### June 13, 2022

- :bulb: larger model gets better CCE performance, see https://ml.azure.com/runs/scFormer-40fd5876-diverse-drake-cellxgene_f10-53ec3928 with model scale of 2.
- [x] Gene vocabulary and default gene vocabulary

### June 9, 2022

- Plan for processing the data: we do want it to be flexible, but let's do some preprocessing and store the processed data in a file first. The anndata objects takes so large memory, it is not convenient to mapp them in the hard drive. Future solution can be, (1) find a way to directly access and slice data on the fly while keeping the studies structure. This means maintain the cell-genes as an large data endpoint while the meta study info as flexible annotation tags. Like a database.
- Add the data structure for large-scale computing.
- [x] add logging to file.
- [x] storing intermediate checkpoints.
- [x] warm up ratio.
- [ ] add an objective to let the model learn to not distinguish the origin study label.

- [ ] Only pretrain non-zero genes could have potential issues, since the model will only predict non-zero gene expression value. (1) This will be an issue if building up a decoder for generation. (2) This could be an issue when seeing new cells and the model would think all input genes have significant values and all gene expressions are learned when it has significant values. A potential **solution** is to include a portion of zero genes to predict in the MLM pretraining task.

### June 8, 2022

- [ ] Memory mapping and streaming for large-scale training data using huggingface datasets. We can probably use the Datasets to load, stream, store in memory, and export most of the tables, or even all the tables together. Useful references:
  - https://huggingface.co/docs/datasets/loading#specify-features
  - specify json fields https://huggingface.co/docs/datasets/loading#json
  - devide into chunks https://huggingface.co/docs/datasets/process#shard
  - save to disk or export to json/parquet https://huggingface.co/docs/datasets/process#save, https://huggingface.co/docs/datasets/process#export
  - https://huggingface.co/docs/datasets/process#flatten
  - label2id mapping https://huggingface.co/docs/datasets/process#align
  - set num_proc, with_indices, batch processing

### June 6, 2022

- [ ] Preprocess large-scale data,
  1. standardize cell type into unified integer labels. tyep str <-> int dictionaries.
  2. the normalization strategy.
  3. standardaaize gene names into a full gene vocabulary.
  4. we can do all self-supervised CCE training first.

### June 4, 2022

- [x] Dev script setup.
- [ ] Sfaira usage setup.
- [ ] Tokenizer setup.
- [ ] Support huggingface interfaces.

### May 29, 2022

- [ ] Set up the new promising evaluation metric.
- [ ] Make sure use model.eval for all inference settings.

### May 26, 2022

- [ ]Sfaira and HCA interface, and download
- [ ]Support data parallelism training, consider integrate huggingface

### May 19, 2022

- [ ] Convert notebooks to test py scripts.
- [ ] How to balance the optimization between contrastive objective and MLM objective?

### May 15, 2022

- [x] Supervised objectives
- New dataset purified PBMC
- [x] Contrastive cell embedding objective
- [ ] Looking at the contrastive learning results: mse 23.67 | mre 0.43 |cce 18.64. The training mse actually increased a lot compared to the only MLM training. This indicate on one hand a good sign that the CCE is making this a challenging task and the model is trying to learn. ON the other hand, we should _increase the model size_ there should be some potential to let the model learn to perform as best as previous ones, since the two tasks should be not interfering with each other.

### May 8, 2022

- [ ] In batch contrastive learning like [SimCSE](https://arxiv.org/abs/2104.08821). positive example: another dropout gene/value negative examples; other examples in batch.
- [ ] Be brave to find more and introduce Supervised information for pretraining.

### May 3, 2022

- Some counts are extremely large (600+ when the avg is 3.x). The choice of normalizing
  the counts and the objective is crucial. Consider
  1. logp1 normalization.
  2. clipping count to 3 delta.
  3. using the absolute error instead of squared error.
- The training objective is extremely important. For example, if set the objective to
  mse, validdation can have mse ~5.0, and relative error ~0.3. However, if set the
  objective to relative error, the validation mse increase to 40+. One reason should
  be the squared ones emphasize more on the larger values.

### May 2, 2022

- The input read value scaling could have significant impacts on the batch effects.
  The gene embedding is fine. Theoretically, if the cell embeddings rely on both
  gene and read value information, the batch effects between values could be the
  issue. So, need clever design to (1) make cell embeddings agnostic to read values;
  (2)remove much batch effects using carefully normalized inputs.
- The normalization of genes could be an issue. So normalizing all cells' expression
  values to the same scale [0, 1] could be problematic, since the overall gene
  expression counts could be important indicator for celltypes. For example, an
  active stem cell could have high expressions on average.
- Check this paper for mormalization comparison https://doi.org/10.1186/s13059-019-1861-6.

### May 1, 2022

- For the Masked Value objective, since (1) the value embedding layer projects the
  value float -> embedding; (2) the output layer projects the embedding -> float.
  Shall we:
  1. Constrain the same embeddings for the input and output layers?
  2. Add a seperate autoencoder at the input layer and copy the decoder (of the
     autoencoder) weights to the output layer? This seems a very strong constraint
     and add new loss terms so the transformer part is also forced to obey the autoencoder ruels.
  3. We could simply use separate output layer weights. Or simply learn mse loss
     for the output to match the emebedding of the correct read value.
  4. Any other paper addresses this?

### April 30, 2022

Carefully read BERT, GPT, Transformer, graphformer, and the relative embedding paper.
Be particularly careful about positional embeddings.

### April 26, 2022

- The esssential steps are:

  1. [x] build the tokenizer
  2. [x] build the transformer
  3. [ ] training and testing logics.

  Currently, we mainly follow the huggingface transformers interfaces and consult codeBert inplementations.

- Other from-scratch transformer implementations: [CodeT5](https://github.com/salesforce/CodeT5),
  [Graphormer](https://github.com/microsoft/Graphormer),
  [CodeBERT](https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/code2nl)

### April 24, 2022

- 2 choices for custom transformer:
  1.  :heart: Huggingface (Support of configs, trainers, optimizers, workflow ...)
  2.  Pytorch (More flexibility)
- 4 choices for building trainer:
  1.  :heart: Huggingface: (trainer, scaling, hyperpara-tuing integrations)
  2.  Pytorch
  3.  :heart: Pytorch Lightning (hydra, deepspeed, ...)
  4.  :heart: Consult scvi-tools (better consistent interface with scanpy)

### April 22, 2022

- dtasets we can use: https://docs.scvi-tools.org/en/stable/api/datasets.html,
  https://scanpy.readthedocs.io/en/stable/api.html#module-scanpy.datasets,
  repo[single cell datasets for python]

### April 20, 2022

- Let's work on applying the single cell transformer on the integration first,
  consult and use this to validate our proposal, and we put this first part into
  a work. Use the code and workflow for running tasks in the existing works, particulaly
  scvi, scJoint, scArches, DEEPMAPS, etc.
