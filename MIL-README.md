# Multiple Instance Learning (MIL) Dataset and Training Code Documentation

## Overview
This documentation provides a detailed explanation of the **Multiple Instance Learning (MIL)** dataset handling and training process using the `MIL_DataSet.py` and `MIL_train.py` scripts. These scripts are designed for processing whole-slide images (WSIs) and training a deep learning model for histopathological image classification. The implementation follows the MIL paradigm, where a slide is labeled rather than individual patches, and training is performed by aggregating top-k highest scoring patches.

## Assumptions
- Input images are stored as pyramidal TIFFs, processed using **OpenSlide**.
- The dataset is stored as a binary **PyTorch library file** containing slide paths, tile locations, and labels.
- The model follows a **ResNet-34** backbone, trained using a cross-entropy loss function.
- Training follows **top-k instance selection**, assuming the k-highest probability patches determine slide classification.

---

# **MIL Dataset (MIL_DataSet.py)**

### **1. Imports & Dependencies**
The dataset script uses:
- **OpenSlide**: Handling Whole-Slide Images (WSIs)
- **PIL (Pillow)**: Image processing
- **PyTorch & Torchvision**: Data handling, transformations, and model utilities
- **NumPy & Random**: Data processing

### **2. MILdataset Class**
The `MILdataset` class extends `torch.utils.data.Dataset` to handle MIL data.

#### **2.1. Initialization (`__init__` method)**
```python
class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None):
```
- Loads a **library file** containing preprocessed WSI metadata.
- Opens WSI headers using OpenSlide.
- Extracts a **grid of tile coordinates** and slide indices.
- Stores:
  - `self.slides`: List of OpenSlide objects for each WSI.
  - `self.grid`: List of patch coordinates extracted from WSIs.
  - `self.slideIDX`: Index mapping between patches and WSIs.
  - `self.targets`: Slide-level classification labels.
  - `self.mult`: Resizing factor for patches.
  - `self.size`: Patch size after resizing (`224 × mult`).
  - `self.level`: Pyramid level to extract tiles from.

#### **2.2. Set Mode (`setmode` method)**
```python
def setmode(self, mode):
    self.mode = mode
```
- **Mode 1**: Inference - Returns only image patches.
- **Mode 2**: Training - Returns image patches and corresponding slide labels.

#### **2.3. Training Data Preparation (`maketraindata` method)**
```python
def maketraindata(self, idxs):
    self.t_data = [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]]) for x in idxs]
```
- Constructs a list of `(slide index, tile coordinates, slide label)` for training.

#### **2.4. Training Data Shuffling (`shuffletraindata` method)**
```python
def shuffletraindata(self):
    self.t_data = random.sample(self.t_data, len(self.t_data))
```
- Randomly shuffles training patches.

#### **2.5. Get Item (`__getitem__` method)**
Handles different modes:
- **Inference mode (mode=1):** Loads patch, applies transformations, returns image.
- **Training mode (mode=2):** Loads patch, applies transformations, returns `(image, target label)`.

#### **2.6. Dataset Length (`__len__` method)**
```python
def __len__(self):
    if self.mode == 1:
        return len(self.grid)
    elif self.mode == 2:
        return len(self.t_data)
```
- Returns the number of available patches depending on the mode.

---

# **MIL Training (MIL_train.py)**

### **1. Command Line Arguments (argparse)**
Arguments include:
- `--train_lib`: Path to training dataset.
- `--val_lib`: Path to validation dataset.
- `--batch_size`: Training batch size.
- `--nepochs`: Number of training epochs.
- `--workers`: Number of data loading threads.
- `--test_every`: Validation frequency.
- `--weights`: Class imbalance weight for loss calculation.
- `--k`: Top-k instance selection strategy.

### **2. Model Initialization**
```python
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.cuda()
```
- Loads **ResNet-34** and replaces the final layer with a **2-class output**.

### **3. Loss Function & Optimizer**
```python
if args.weights == 0.5:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    w = torch.Tensor([1-args.weights, args.weights])
    criterion = nn.CrossEntropyLoss(w).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
```
- Uses **CrossEntropyLoss**, optionally weighted for class imbalance.
- Optimized using **Adam** with **learning rate=1e-4**.

### **4. Data Loading**
- Training and validation datasets are loaded using `MILdataset` and DataLoader.

### **5. Training Loop**
```python
for epoch in range(args.nepochs):
    train_dset.setmode(1)
    probs = inference(epoch, train_loader, model)
    topk = group_argtopk(np.array(train_dset.slideIDX), probs, args.k)
    train_dset.maketraindata(topk)
    train_dset.shuffletraindata()
    train_dset.setmode(2)
    loss = train(epoch, train_loader, model, criterion, optimizer)
```
- Performs inference on all patches to generate probabilities.
- Selects **top-k patches per slide** for training.
- Trains the model with the selected patches.

### **6. Inference Function**
```python
def inference(run, loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()
```
- Runs forward pass on all patches, computing **softmax probabilities**.
- Stores probabilities for **top-k instance selection**.

### **7. Error Calculation**
```python
def calc_err(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    err = float(np.not_equal(pred, real).sum()) / pred.shape[0]
    fpr = float(np.logical_and(pred==1, np.not_equal(pred, real)).sum()) / (real==0).sum()
    fnr = float(np.logical_and(pred==0, np.not_equal(pred, real)).sum()) / (real==1).sum()
    return err, fpr, fnr
```
- Computes **error rate, false positive rate (FPR), and false negative rate (FNR)**.

### **8. Model Checkpointing**
- Saves the best-performing model based on the lowest error rate.


