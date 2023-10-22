# What? Why?
This repository contains code for using [GROOViST: A Metric for Grounding Objects in Visual Storytelling]() — In proceedings at the EMNLP 2023 (*To appear*).

Evaluating the degree to which textual stories are grounded in corresponding image sequences is essential for the Visual Storytelling task. We propose GROOViST, based on insights obtained from existing open-source metrics ([CLIPScore](https://github.com/jmhessel/clipscore), [RoViST-VG](https://github.com/usydnlp/rovist)). Our analyses shows that GROOViST effectively measures the extent to which a story is grounded in an image sequence.

If you find this work useful, please consider citing it:
```
<TODO: update bibtex>
```

# How?

Currently, GROOViST can be used off-the-shelf for evaluating `<image-sequence, story>` pairs of three Visual Storytelling datasets — [VIST](https://aclanthology.org/N16-1147/), [AESOP](https://openaccess.thecvf.com/content/ICCV2021/html/Ravi_AESOP_Abstract_Encoding_of_Stories_Objects_and_Pictures_ICCV_2021_paper.html), [VWP](https://aclanthology.org/2023.tacl-1.33/). For a new/custom dataset, all the following steps can be adapted accordingly.

## Setup

Install python (e.g., `3.11`) and other dependencies provided in [requirements.txt](requirements.txt), e.g., using `pip install -r requirements.txt`

## Step 0: Extract image regions

For the sequence(s) of interest, GROOViST requires `B` image regions per image in the sequence(s) (e.g., `B=10`). Please refer to [this doc](image_regions.md) for preparing them.

## Step 1: Extract noun phrases

For the sequence(s) of interest, GROOViST works with the noun phrases in the stories. Use the following command for extracting noun phrases from stories:
```python extract_nphrases.py --input_file data/sample_stories.json --output_file data/sample_nphrases.json```

## Step 2: Compute GROOViST scores

```python groovist.py --dataset VIST --input_file data/sample_nphrases.json --output_file data/sample_scores.json```
