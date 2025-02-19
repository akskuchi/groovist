[![CC BY license](https://img.shields.io/badge/License-CC%20BY-lightgray.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/python-3.11-gold.svg)](https://www.python.org/downloads/release/python-311/)
[![PyTorch](https://img.shields.io/badge/Pytorch-2.0-pumpkin.svg)](https://pytorch.org/get-started/previous-versions/#v200)

# 👀 What?
This repository contains code for using **[GROOViST: A Metric for Grounding Objects in Visual Storytelling](https://aclanthology.org/2023.emnlp-main.202/)**&mdash;In proceedings of EMNLP 2023.

# 🤔 Why?
Evaluating the degree to which textual stories are grounded in the corresponding image sequences is essential for the Visual Storytelling task. We propose GROOViST, based on insights obtained from existing open-source metrics ([CLIPScore](https://github.com/jmhessel/clipscore), [RoViST-VG](https://github.com/usydnlp/rovist)). Our analyses shows that GROOViST effectively measures the extent to which a story is grounded in an image sequence.

# 🤖 How?
Currently, GROOViST can be used off-the-shelf for evaluating `<image-sequence, story>` pairs of three Visual Storytelling datasets — [VIST](https://aclanthology.org/N16-1147/), [AESOP](https://openaccess.thecvf.com/content/ICCV2021/html/Ravi_AESOP_Abstract_Encoding_of_Stories_Objects_and_Pictures_ICCV_2021_paper.html), [VWP](https://aclanthology.org/2023.tacl-1.33/). For a new/custom dataset, all the following steps can be adapted accordingly.

## Setup
Install python (e.g., `3.11`) and other dependencies provided in [requirements.txt](requirements.txt). E.g., using:

```pip install -r requirements.txt```

## Step 0: Extract image regions
For the sequence(s) of interest, GROOViST requires `B` image regions per image in the sequence(s) (e.g., `B=10`). Please refer to [this doc](image_regions.md) for preparing them.

## Step 1: Extract noun phrases
For the sequence(s) of interest, GROOViST works with the noun phrases in the stories. Use the following command for extracting noun phrases from stories:

```python extract_nphrases.py --input_file data/sample_stories.json --output_file data/sample_nphrases.json```

## Step 2: Compute GROOViST scores
```python groovist.py --dataset VIST --input_file data/sample_nphrases.json --output_file data/sample_scores.json```

---
🔗 If you find this work useful, please consider citing it:
```
@inproceedings{surikuchi-etal-2023-groovist,
    title = "{GROOV}i{ST}: A Metric for Grounding Objects in Visual Storytelling",
    author = "Surikuchi, Aditya K  and
      Pezzelle, Sandro  and
      Fern{\'a}ndez, Raquel",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.202/",
    doi = "10.18653/v1/2023.emnlp-main.202",
    pages = "3331--3339"
}
```
