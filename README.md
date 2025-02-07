# MIL-Camelyon

This repository prepares the **Camelyon16** dataset for **Weakly Supervised Learning** under the **Multiple-Instance-Learning (MIL)** paradigm. Each image is at gigapixel scale, so images must be tiled into target sizes depending on the network architecture (e.g., **256x256** for ResNets). Background tiles are discarded by performing **tissue segmentation**, generating a binary mask, and then selecting tile coordinates accordingly.

## Tiling Pipeline

In **`prep_tiles.ipynb`**, we:
- Extract all slide paths in the dataset.
- Apply tiling to the images.
- Store tile coordinates for each dataset as a list of lists:

```python
grid = [
    [(x1_1, y1_1), (x1_2, y1_2), (x1_3, y1_3)],
    [(x2_1, y2_1), (x2_2, y2_2), (x2_3, y2_3), (x2_4, y2_4)]
]
```

- The size of the outer list is equal to the number of images.
- The size of the inner lists are equal to the number of tiles per image.

## Tissue Segmentation

Tissue segmentation is performed without neural networks or manual annotations. Instead, tissue is separated from the background using:
1. Gradient magnitude calculation across the image.
2. Otsu's thresholding to discard background tiles.

![image](https://github.com/user-attachments/assets/7efba910-4d16-4dfb-abdc-786350c5bfc4)


A demonstration is available in **`tiling_demo.ipynb`**. These functions are stored in **`wsi_prep.py`** for use in data preprocessing.

## Image Tiling

Since loading whole-slide images at full resolution (level 0) is computationally expensive, they are instead loaded at a downsampled factor available within the pyramidal `.tif` structure of the images. 

### Steps:
- The downsampled patch size is calculated from the downsample factor at the level the image is loaded at and the desired patch size at level 0 (ie, 256).
- Images are **tiled without overlap** at the downsampled level.
- Tile coordinates are **mapped back to level 0**.

![image](https://github.com/user-attachments/assets/c14875c2-8c65-4d3a-aacf-0af45c600ce2)


A demonstration of the tiling process can be found in **`tiling_demo.py`**.

