# MIL-Camelyon

This repository prepares the **Camelyon16** dataset for **Weakly Supervised Learning** under the **Multiple-Instance-Learning (MIL)** paradigm. Each image is at gigapixel scale, so images must be tiled into target sizes depending on the network architecture (e.g., **256x256** for ResNets). Background tiles are discarded by performing **tissue segmentation**, generating a binary mask, and then selecting tile coordinates accordingly. 

## Multiple Instance Learning 

### Problem Formulation
In MIL, data is organized into bags of instances. Each bag corresponds to a WSI, and the instances within the bag are tiles extracted from that slide. The key assumption in MIL is that:
  - A positive bag contains at least one positive instance.
  - A negative bag contains only negative instances.

### Mathematical Representation
Let:

$$
\mathcal{B} = \{B_1, B_2, \dots, B_N\}
$$
represent a set of bags.

Each bag is defined as:

$$
B_i = \{x_{i1}, x_{i2}, \dots, x_{im_i}\}
$$
where it contains instances.

The label for bag \( B_i \) is given by \( y_i \), where:

- \( y_i = 1 \) indicates a positive bag.
- \( y_i = 0 \) indicates a negative bag.

Instance-level labels \( z_{ij} \) are unknown.

The relationship between bag labels and instance labels can be expressed as:

$$
y_i =
\begin{cases}
1 & \text{if } \exists j : z_{ij} = 1, \\
0 & \text{if } z_{ij} = 0, \forall j.
\end{cases}
$$



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

## MIL Training

The saved tile coordinates are then used to create a training dictionary with the structure:

```python
dict ={ 

    "slides": #full paths to WSIs
    "grid": #see above
    "targets": #list of slide level class
    "mult": #scale factor if desired resoltion is different to what is saved in the tiff
    "level": #WSi pyrmaid level from which tiles should be read (usually 0)

}
```

This dictionary is then used to create a custom Pytorch DataSet (see **`MIL_DataSet.py`**), which is then used in **`MIL_train.py`**, which trains an instance level representation, to be later used in an aggregator for bag level classification. 

