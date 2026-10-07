# Data terms

The MIT License in [LICENSE](LICENSE) covers the source code only. It does not cover the data: the
files in `data/` and `figures/`, and the embeddings file `cub200_cls_embeddings.npy` (attached to the
GitHub release v1.0.0 and published as the Hugging Face dataset
`ricardojmveiga/cub200-dinov3-7b-embeddings`). The data are derived from the three sources below,
whose terms continue to apply.

## 1. CUB-200-2011

C. Wah, S. Branson, P. Welinder, P. Perona and S. Belongie, "The Caltech-UCSD Birds-200-2011
Dataset", Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.
<https://www.vision.caltech.edu/datasets/cub_200_2011/>, <https://doi.org/10.22002/D1.20098>.

No images are included. `data/cub200_image_paths.json` lists the dataset's image file names (from
`images.txt`) and `data/cub200_species_names.json` its class names (from `classes.txt`). The
dataset's authors state that they do not own the copyrights to the images and that the images' use is
restricted to non-commercial research and educational purposes. The embeddings, the species
centroids and the clustering results are computed from those images, so use them only for
non-commercial research and educational purposes.

## 2. DINOv3

The features were computed with the frozen model `facebook/dinov3-vit7b16-pretrain-lvd1689m`
(O. Siméoni et al., "DINOv3", arXiv:2508.10104, 2025), which Meta distributes under the DINOv3
License (<https://ai.meta.com/resources/models-and-libraries/dinov3-license/>; a copy is in
[third_party/DINOv3_LICENSE.md](third_party/DINOv3_LICENSE.md)). No model weights or model code are
included.

**Built with DINOv3.**

## 3. Taxonomy

The order, family and genus of each class come from the Wikispecies-derived CUB-200-2011 hierarchy
of B. Barz and J. Denzler (<https://github.com/cvjena/semantic-embeddings>, folder `CUB-Hierarchy`;
please cite B. Barz and J. Denzler, "Deep Learning on Small Datasets without Pre-Training using
Cosine Loss", WACV 2020, as that folder asks). It is distributed under the MIT License, Copyright (c)
2020 Björn Barz; a copy is in
[third_party/semantic-embeddings_LICENSE.txt](third_party/semantic-embeddings_LICENSE.txt).

## The authors' contribution

Subject to the terms above, the authors license their own contribution to the data under the
Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0,
<https://creativecommons.org/licenses/by-nc/4.0/>). Copyright (c) 2026 Ricardo J. M. Veiga and
João M. F. Rodrigues. The data are provided as is, without warranty of any kind.
