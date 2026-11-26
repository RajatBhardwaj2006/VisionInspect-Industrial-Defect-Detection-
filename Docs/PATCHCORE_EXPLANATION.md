# PatchCore in VisionInspect

PatchCore models what a normal product looks like at many local positions. A
scratch can affect only a small part of an otherwise normal image, so comparing
local feature vectors is useful for both anomaly detection and localization.
Training uses normal images; defective images and their masks are for evaluation.

This is an explanation of the implementation in
[`patchcore_v22.py`](../src/models/patchcore_v22.py) and the production
[`v2.3 detector`](../src/detection/patchcore_v23_detector.py). It does not prescribe
changes to the shipped checkpoints or calibration.

## ResNet18 and multi-scale features

The frozen ImageNet-pretrained ResNet18 backbone extracts features rather than
predicting an ImageNet class. For a batch of 256 x 256 RGB inputs:

| Feature | Shape per image | Role |
| --- | --- | --- |
| Layer 1 | 64 channels, 64 x 64 | Fine surface and edge detail |
| Layer 2 | 128 channels, 32 x 32 | Intermediate patterns |
| Layer 3 | 256 channels, 16 x 16 | Larger structures |

Layers 2 and 3 are bilinearly interpolated to the layer 1 grid. Concatenating the
channels gives **64 + 128 + 256 = 448** features at each of **4,096** locations.
The extractor returns shape `[batch, 4096, 448]` and grid shape `(64, 64)`.
The grid follows the input resolution: a 224 x 224 input would not produce the
same 64 x 64 grid. These locations have overlapping receptive fields; they are
not independent 4 x 4 image crops.

## Normal memory bank

During `fit()`, embeddings from normal training images are flattened into a large
collection. The current implementation selects a random subset using
`torch.randperm`, with a default sampling ratio of 0.10. This reduces storage and
nearest-neighbour cost. Although called a coreset in project descriptions, this
particular implementation does **not** implement minimax greedy selection.
Production inference loads an existing bank; it does not rebuild it per upload.

## Distances and anomaly maps

For patch vector `z_i` and memory bank `M`, the local distance is:

```text
d_i = min over m in M of ||z_i - m||_2
```

A patch close to normal training vectors has a small distance. A distant patch
is a candidate anomaly. Search uses `torch.cdist` in chunks of 512 query patches
to bound the temporary distance matrix. The distances form a 64 x 64 map, then
bilinear interpolation expands it to the working image resolution.

Where enabled, a spatial prior is subtracted and the map is clamped at zero:

```text
A_corrected(x, y) = max(A_raw(x, y) - P(x, y), 0)
```

The model supports a mean normal map and an available 75th-percentile (`p75`)
normal map. Both are reference maps from normal training data. Which prior is
used depends on category configuration and available artifacts; textures such
as leather do not use a fixed spatial prior. Gaussian smoothing follows in the
detector.

## Image score versus localization

The model returns the maximum raw patch distance without a prior, or the maximum
corrected map value when a prior is applied. The detector uses that score for
the default `max_raw` path. For transistor (`top200`) and screw (`top100`), it
instead averages the largest 200 or 100 values of the **smoothed 256 x 256 map**.
These are map pixels, not 200 or 100 original feature vectors.

The ordinary classification compares the image score with its calibrated image
threshold. Pixel thresholding and morphology then identify regions to explain
where the defect lies. Border filtering, merging and fallback recovery affect
the boxes. Transistor's presence check can override the ordinary score verdict.
The [architecture guide](PROJECT_ARCHITECTURE.md) describes the full runtime path.

Heatmap colors visualize relative spatial evidence. Neither the distances nor
their colors are defect probabilities, and scores from different category banks
should not be compared as a universal severity scale. Use the selected category's
threshold and explanation when interpreting an inspection.
