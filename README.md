# HCD Lymph Node Cancer Assistant

> Binary classification of histopathologic H&E image patches from lymph-node tissue into **metastatic** vs **non-metastatic**.

---

## 1. Project Overview

This project builds an end-to-end pipeline for metastasis detection on lymph-node H&E patches using the **Kaggle Histopathologic Cancer Detection (HCD)** dataset.  
Our goals:

- Reproduce and extend classic baselines (ResNet-18 / ResNet-34) on a **stratified** data split.
- Study how **stain normalization** (Macenko) and **color augmentation** affect model performance.
- Replace the traditional ResNet backbone with a modern **EfficientNetV2-S** classifier.
- Deploy the best model as a **web-based assistant** with Grad-CAM heatmaps and Chinese AI explanations.

The final system can take a single H&E patch as input, output a cancer probability, visualize model attention, and generate a short human-readable explanation of the prediction. :contentReference[oaicite:0]{index=0}  

---

## 2. Dataset (Not Included in This Repository)

We use the public **Histopathologic Cancer Detection** (HCD) dataset from Kaggle:

- ~220k RGB patches, each **96×96** pixels.
- Each patch is cropped from a lymph-node whole-slide image.
- Label = whether the **central region** of the patch contains metastatic tumor (1) or not (0).

Because the dataset is quite large, **raw images are not stored in this GitHub repo**.  
To reproduce the experiments, please download the dataset directly from Kaggle:

- Kaggle competition: *Histopathologic Cancer Detection*  
  (search on Kaggle or follow the competition page link).

---

## 3. Data Understanding

We first performed exploratory analysis on the original training set:

- **Class imbalance**: the positive (metastatic) class is less frequent than the negative class.
- **Color and stain variation**:
  - Hue alone is not very discriminative.
  - Brightness and saturation show noticeable shifts between patches.
  - RGB channel distributions reveal stain-intensity variation and slide-to-slide bias.

These observations motivate stain normalization and carefully designed color augmentation rather than using the raw images directly. :contentReference[oaicite:1]{index=1}  

---

## 4. Data Preparation

### 4.1 Stratified Split

Instead of the original paper’s random 8:1:1 split, we create a **stratified 70 / 15 / 15 train–val–test split**, maintaining similar positive / negative ratios in each subset.  
This leads to more stable AUROC estimates and fairer comparison between models.

### 4.2 Stain Normalization (Macenko)

We adopt **Macenko stain normalization** to reduce stain-intensity variation between slides:

1. Convert RGB to optical density (OD) space.
2. Estimate two stain basis vectors for Hematoxylin and Eosin.
3. Project each patch onto this basis and align stain intensities to a reference.
4. Reconstruct normalized RGB patches with more consistent stain colors.

### 4.3 2×2 Color–Stain Design

To systematically study the effect of stain normalization and color augmentation, we use a **2×2 experimental design**:

- Raw HCD patches vs **Macenko-normalized** patches.
- **With** vs **without** color augmentation.

All images are resized to **256×256** and normalized before feeding into the network.

### 4.4 Color Augmentation Strategy

- **Raw HCD**:
  - Moderate brightness/contrast and gamma changes.
  - Occasional small RGB shifts.
  - A small portion of patches kept nearly unchanged to preserve realistic colors.

- **Macenko-normalized**:
  - Already standardized, so only gentle brightness/contrast and gamma variation.

The aim is to simulate realistic lab-to-lab stain differences while avoiding unrealistic colors that might hurt generalization.

---

## 5. Modeling

### 5.1 Baseline and Reference Models

We re-implement the main models from a previous HCD study on our stratified split:

- **ResNet-18** baseline (ImageNet-pretrained, full fine-tuning).
- **ResNet-34** as a strong reference (paper’s reported SOTA on the original split).

The training recipe largely follows the reference work (random crops, horizontal/vertical flips, ImageNet normalization), adapted to our split.

### 5.2 Proposed Model: EfficientNetV2-S

Our main model is an **EfficientNetV2-S** patch classifier:

- Input: 256×256 H&E patch.
- Backbone: EfficientNetV2-S (ImageNet-1k pretrained).
- Head: global average pooling → fully connected layer → sigmoid.
- Loss: class-balanced **BCEWithLogitsLoss** to up-weight metastatic patches.
- Optimization:
  - AdamW optimizer with cosine learning-rate schedule.
  - Mixed-precision (AMP) training on an 8-GPU server (DDP).
  - Approx. 12 epochs with a large effective batch size.

The model outputs a continuous **cancer probability** for each patch.

---

## 6. Experiments & Results

### 6.1 Comparison with ResNet Baselines

On the same stratified 70/15/15 split:

- Our re-implemented **ResNet-18** baseline reproduces the expected performance from the reference work.
- **ResNet-34** (the reference SOTA) performs better than ResNet-18, but is still outperformed by EfficientNetV2-S on AUROC and F1.

### 6.2 Effect of Stain Normalization and Color Augmentation

Key findings from the 2×2 design (Raw vs Macenko × Color Aug ON/OFF): :contentReference[oaicite:2]{index=2}  

- **Raw H&E + color augmentation** gives the **best overall performance**.
- Color augmentation on raw HCD provides a **small but consistent improvement in F1-score**.
- **Macenko stain normalization** slightly **reduces performance** in this setting:
  - It successfully reduces stain variation, but seems to remove some subtle cues that the model can exploit.
  - Gentle augmentation on Macenko-normalized images cannot fully close the gap to raw HCD.

### 6.3 Final Model Performance (Qualitative Summary)

The best configuration is:

- **EfficientNetV2-S** trained on **raw H&E patches with color augmentation**.

Compared with the strong ResNet-34 reference:

- **AUROC**: EfficientNetV2-S achieves the highest AUROC among all tested models.
- **F1-score**: also the highest, indicating a good balance between sensitivity and specificity.
- **Recall**: improves from roughly **0.94** (ResNet-based baselines) to about **0.97**, which is particularly important for **not missing metastatic patches**. :contentReference[oaicite:3]{index=3}  

Overall, a modern, parameter-efficient backbone plus carefully tuned color augmentation yields clear improvements over the classic ResNet baselines.

---

## 7. Web Deployment: HCD Lymph Node Cancer Assistant

We deploy the best EfficientNetV2-S model as a simple **web assistant**:

- **Backend**:
  - FastAPI application.
  - EfficientNetV2-S model for patch-level prediction.
  - Grad-CAM to visualize regions that most influence the cancer prediction.

- **Frontend workflow**:
  1. Upload a lymph-node H&E patch (single image).
  2. The app preprocesses the image (resize + normalization).
  3. The model outputs:
     - Cancer probability (0–1).
     - A binary label: **Positive (cancer)** vs **Negative (benign)**.
     - A **Grad-CAM heatmap** overlaid on the patch.
  4. Optional: click “Generate AI Explanation (beta)” to get a short **Chinese explanation** that describes:
     - The predicted risk level.
     - The approximate hotspot region in the patch.
     - A reminder that this is a research tool and not a diagnostic device. :contentReference[oaicite:4]{index=4}  

This demo illustrates how a patch-level classifier can be integrated into a more user-friendly interface for education and research.

---

## 8. Limitations & Future Work

**Limitations**

- Only a single public dataset (HCD) is used, with one stratified 70/15/15 split.
- No external validation on other hospitals or stain protocols.
- Model families are restricted to supervised **ResNet18/34** and **EfficientNetV2-S**.
- The system operates at **patch level**; it does not perform slide-level aggregation or detection.

**Future directions**

- Explore **self-supervised pretraining** and more stain-robust architectures.
- Evaluate on multiple datasets and external cohorts to test generalization.
- Extend from patch-level classification to **slide-level** metastasis detection.
- Integrate clinical metadata and multi-scale context for more realistic decision support.

---

## 9. Acknowledgements

- Dataset: Kaggle **Histopathologic Cancer Detection** competition.
- Reference baselines: ResNet architectures and training recipes from prior HCD work.
- This project was completed as part of a data science course, including a final presentation and demo web app.
