# CORA: Consistency-Guided Semi-Supervised Framework for Reasoning Segmentation

Code for training and evaluating CORA ([CORA:WACV2026]([https://arxiv.org/abs/2511.17755](https://openaccess.thecvf.com/content/WACV2026/papers/Howlader_CORA_Consistency-Guided_Semi-Supervised_Framework_for_Reasoning_Segmentation_WACV_2026_paper.pdf))). 

CORA is a reasoning-segmentation model in the style of LISA: a multimodal LLM (LLaVA) emits a `<SSEG>` token whose
hidden state prompts the SAM mask decoder. It is trained with a small labeled fraction of Cityscapes, instruction sets
built on the ground-truth masks of those images, and the remaining images as unlabeled data with pseudo-labels from a
semi-supervised segmenter. For unlabeled images the pseudo-label loss of every pixel is weighted by how consistent the
model's own predictions are across rephrasings of the same query, and the `<SSEG>` features of unlabeled images are
aligned with those of labeled images of the same class.

### Contact
If you have any questions, please email Prantik Howlader at phowlader@cs.stonybrook.edu.

## What this repository trains

| | |
|---|---|
| Starting weights | the released LISA-7B-v1 checkpoint (`xinlai/LISA-7B-v1`), converted by `tools/convert_lisa.py` |
| Trained parts | LoRA (rank 8, `q_proj` and `v_proj`) and the SAM mask decoder; from stage 2 on also the anchor encoders. Everything else is frozen |
| Stage 1  | labeled images: semantic and attribute instruction sets |
| Stage 2  | labeled images: conditional instruction set, with the anchor mask and box encoders |
| Stage 3 | all labeled sets plus the unlabeled images: pseudo-label loss with per-pixel consistency weights over K = 7 rephrasings, summed over pixels with weight 0.001, and the token-level contrastive alignment |
| Pseudo-labels | SemiVL, the public checkpoint of each partition |
| ReasonSeg | 1 labeled sample in 4 is drawn from the ReasonSeg train split, in every stage |
| Optimisation | AdamW, learning rate 2e-5 with linear warm-up and decay, batch 8 (2 images on each of 4 GPUs) |

The four configurations are `configs/cityscapes/{1_30,1_16,1_8,1_4}.yaml`; they differ only in the labeled fraction.
`configs/cityscapes/base.yaml` holds the shared settings and `cora/config.py` documents every option.

## Requirements

* Linux with NVIDIA GPUs. Training was run on 4 GPUs with 40 GB each (about 23 GB used on each, about 5.5 hours per
  partition); with fewer GPUs keep the batch of 8 with `train.grad_accum_steps` (2 GPUs: 2, 1 GPU: 4).
* conda (the scripts create two environments).
* About 80 GB of disk: Cityscapes (12 GB), the LISA checkpoint and its converted copy (16 GB each), the SemiVL
  checkpoints and environment, pseudo-labels and instruction sets.

## Installation

```bash
git clone <this repository> CORA && cd CORA
bash scripts/create_env.sh        # conda environment in envs/cora: Python 3.11, torch 2.11.0 (CUDA 12.8), requirements.txt
source scripts/env.sh             # activates it and sets CORA_ROOT and PYTHONPATH
```

`CORA_ROOT` is the folder that holds `data/`, `datasets/`, `pretrained/`, `outputs/`, `envs/` and `third_party/`; it
defaults to the repository folder. Set it before `source scripts/env.sh` to keep them elsewhere. The exact package
versions are pinned in `requirements.txt`.

SemiVL needs its own legacy environment (Python 3.8, torch 1.10), which is used only to write the pseudo-labels:

```bash
bash scripts/pseudo/setup_semivl_env.sh                       # clones SemiVL into third_party/ and creates envs/semivl
bash scripts/pseudo/download_teacher_checkpoints.sh semivl    # the four public SemiVL checkpoints
```

## Data and starting weights

**Cityscapes.** Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from
[cityscapes-dataset.com](https://www.cityscapes-dataset.com/) and unpack them into `datasets/cityscapes/`
(`leftImg8bit/`, `gtFine/`), or point `paths.cityscapes_root` to them. The labeled / unlabeled lists of the four
partitions are in `assets/splits/cityscapes/`.

**LISA-7B-v1.** Download the checkpoint, the CLIP weights and the tokenizer, then convert:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("xinlai/LISA-7B-v1", local_dir="downloads/LISA-7B-v1")
snapshot_download("openai/clip-vit-large-patch14", local_dir="downloads/clip", allow_patterns=["config.json", "model.safetensors"])
snapshot_download("llava-hf/llava-1.5-7b-hf", allow_patterns=["*.json", "tokenizer*"])   # tokenizer only
PY
python tools/convert_lisa.py --lisa downloads/LISA-7B-v1 --clip downloads/clip --out pretrained/lisa-7b-v1
```

**ReasonSeg.** Put the dataset released with [LISA](https://github.com/dvlab-research/LISA) into
`datasets/reasonseg/ReasonSeg/{train,val,test}/` (the `.jpg` and `.json` files), or point `paths.reasonseg_root` to
it. A parquet copy (`reasonseg_<split>/data/*.parquet`) is read as well; it needs `paths.reasonseg_manifest`, a JSON
with the long/short flag of every image. Then unpack the train split for training:

```bash
python tools/prepare_reasonseg.py --split train               # writes data/reasonseg/train/
```

**Instruction sets.** The four instruction sets (semantic, attribute, conditional, rule-based conditional) are in
`data/cityscapes/instructions/{train,val}/`: for the 815 training images that are labeled in at least one partition and
for the validation images. `data/cityscapes/README.md` describes the format. Training reads only the records of the
partition's labeled images.

**Masks.** The masks that the instructions refer to are region maps of the Cityscapes annotations. They are not
redistributed here; build them from your Cityscapes copy (a few minutes, identical for everyone):

```bash
for p in 1_30 1_16 1_8 1_4; do python tools/build_objects.py --partition $p --which labeled --workers 8; done
python tools/build_objects.py --split val --workers 8
```

## Running one partition

The example is 1/16; use `1_30`, `1_8` or `1_4` for the others. Every step can be repeated: finished work is kept.
All tools accept `--help`.

**1. Pseudo-labels of the unlabeled images**

```bash
python tools/make_pseudo_labels.py --method semivl --partition 1_16 --date $(date +%F) --gpus 0,1,2,3 \
    --evaluate --report outputs/pseudo_reports/semivl_1_16_val.json        # the teacher's mIoU on the validation images
python tools/make_pseudo_labels.py --method semivl --partition 1_16 --date $(date +%F) --gpus 0,1,2,3 \
    --eval-report outputs/pseudo_reports/semivl_1_16_val.json              # writes data/cityscapes/pseudo/semivl/1_16/
```

The first command checks the teacher before hours of labeling are spent on a wrong checkpoint. UniMatch pseudo-labels
can be used instead (`--method unimatch` after `bash scripts/pseudo/download_teacher_checkpoints.sh unimatch`, then
train with `data.ssl_method=unimatch`); its public checkpoints cover 1/16, 1/8 and 1/4.

**2. Training**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh configs/cityscapes/1_16.yaml 4
```

The three stages run one after the other and write to `outputs/cityscapes_1_16/`. A run that is interrupted
continues from its last checkpoint when the command is repeated. `python tools/train.py --config
configs/cityscapes/1_16.yaml --check` verifies the data and settings without building the model.

**3. Evaluation**

```bash
bash scripts/evaluate.sh configs/cityscapes/1_16.yaml --num-gpus 4                    # Cityscapes validation sets
```

Results are printed as cIoU and gIoU per instruction set and saved under `outputs/cityscapes_1_16/eval/`.
`tools/visualize.py` draws predictions next to the ground truth.

## Repository layout

```
cora/         the package: configuration, data and instruction sets, model, losses, pseudo-labels, training engine
tools/        command-line tools (train, evaluate, convert_lisa, prepare_reasonseg, build_objects, make_pseudo_labels,
              visualize; generate_rule_sets rebuilds the two rule-based instruction sets)
scripts/      launchers for training, evaluation and the two environments
configs/      the four Cityscapes configurations and their shared base
assets/       the partition lists and the query database
data/         the instruction sets for Cityscapes (data/cityscapes/README.md)
tests/        unit tests (CPU, tiny model):  python -m pytest -q
```




## Citation

```bibtex
@InProceedings{Howlader_2026_WACV,
    author    = {Howlader, Prantik and Nguyen-Canh, Hoang and Das, Srijan and Xu, Jingyi and Le, Hieu and Samaras, Dimitris},
    title     = {CORA: Consistency-Guided Semi-Supervised Framework for Reasoning Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {5934-5944}
}
```


