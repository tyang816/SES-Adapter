# Simple, Efficient and Scalable Structure-aware Adapter Boosts Protein Language Models

## 🚀 Introduction (SES-Adapter)

SES-Adapter, a simple, efficient, and scalable adapter method for enhancing the representation learning of protein language models (PLMs). 

We serialized the protein structure and performed cross-modal-attention with PLM embeddings, effectively improving downstream task performance and convergence efficiency.

<img src="img/framework.png" alt="Logo">

## 📑 Results

### Paper Results

We conduct evaluation on 9 state-of-the-art baseline models (**ESM2, ProtBert, ProtT5, Ankh**) across 9 datasets under 4 tasks (**Localization, Function, Solubility, Annotation**).

Results show that compared to vanilla PLMs, SES-Adapter improves downstream task performance by a maximum of **11%** and an average of **3%**, with significantly accelerated training speed by a maximum of **1034%** and an average of **362%,** the convergence rate is also improved by approximately **2** times.

## 🛫 Requirement

### Conda Enviroment

Please make sure you have installed **[Anaconda3](https://www.anaconda.com/download)** or **[Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)**.

```
conda env create -f environment.yaml
conda activate ses_adapter
```

### Hardware

We recommend a **24GB** RTX 3090 or better, but it mainly depends on which PLM you choose.

## 🧬 Start with SES-Adapter

### Use The Provided Dataset

We provide datasets and format references in the `dataset` folder.

### Prepare Your Own Datset

### Train

See the `train.py` script for training details. Examples can be found in `scripts` folder.

## 🙌 Citation

Please cite our work if you have used our code or data.

```
@article{tan2024ses-adapter,
  title={Simple, Efficient and Scalable Structure-aware Adapter Boosts Protein Language Models},
  author={Tan, Yang and Li, Mingchen and Zhou, Bingxin and Zhong, Bozitao and Zheng, Lirong and Tan, Pan and Zhou, Ziyi and Yu, Huiqun and Fan, Guisheng and Hong, Liang},
  journal={arXiv preprint arXiv:2404.14850},
  year={2024}
}
```

