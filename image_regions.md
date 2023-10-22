## I. Image regions

For the VIST dataset, extracted image regions can be obtained from [RoViST-VG](https://drive.google.com/file/d/14lJHWhIQbc0D2khXubz9VyhKIhg57f3-/view?usp=sharing).

For AESOP, VWP, and other custom datasets, image regions can be extracted using the FasterRCNN model (with ResNet-101 backbone) trained on [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) data - [code](https://github.com/airsplay/py-bottom-up-attention).

## II. Mapping between image IDs and extracted image regions

For evaluating the sequence(s) of interest, a mapping between the corresponding `image-id`s and the extracted image region bounding boxes is needed for the metric. For the three visual storytelling datasets, the mapping is available at the respective links:
- VIST: [mapping info file](https://drive.google.com/file/d/1EyA2VNwokV1DNuU2EmV6zrozD9Q8Vgg-/view)
- AESOP (test set only): [mapping info file](data/aesop_entities.csv)
- VWP: [mapping info file](data/vwp_entities.csv)

For new/custom datasets, a similar mapping file can be created by leveraging information during the image regions extraction step.

## III. Mapping between story/scene IDs and image IDs

For connecting sequences to corresponding images, a mapping between story/scene ids and respective image ids is needed for the metric. For VIST and VWP datasets, the mapping is available at the respective links:
- VIST: [story id to image ids](./data/vist_sid_2_iids.json)
- AESOP: not required - since all sequences are made up of 3 images and all image ids follow a defined namespace.
- VWP: [story id to image ids](./data/vwp_scene_id_2_nimgs.json)

**After obtaining the data needed for `I, II,` and `III`, make necessary changes to the [configuration](config.ini) file.**
