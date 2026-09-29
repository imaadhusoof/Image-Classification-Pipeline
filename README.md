# Image Classification Pipeline

An image classification pipeline built for a university machine learning assignment. It compares
hand-crafted image features (colour histograms, HOG + PCA, and more) with deep features taken from
a pretrained **ResNet50**, then trains several classical classifiers and a small neural network on each.

The full pipeline is in [`image_classification_pipeline.ipynb`](image_classification_pipeline.ipynb).
GitHub renders it with all outputs and confusion matrices.

## Tasks

| Task | Dataset | Classes | Train / Test images |
|------|---------|---------|---------------------|
| 1 | Animals (bird, butterfly, cat, deer, dog, elephant, frog, horse, sheep, spider) | 10 | 3,750 / 1,250 |
| 2 | Fine-grained birds (Blue Jay, Cardinal, Herring Gull, Song Sparrow, …) | 10 | 417 / 180 |

## Approach

1. **Baseline on the provided features**: merge colour histogram, HOG-PCA and additional features,
   standardise them, then train KNN, Logistic Regression and an RBF SVM.
2. **Deep feature extraction**: run every image through ImageNet-pretrained ResNet50 with the final FC
   layer removed, which gives a 2048-dimensional embedding per image (uses the GPU if one is available).
3. **Classifiers on the embeddings**: KNN, Logistic Regression, RBF SVM (tuned with `GridSearchCV`),
   Linear SVM, and a PyTorch MLP.
4. **Evaluation**: accuracy, balanced accuracy, macro precision, recall/sensitivity, specificity, macro
   and weighted F1, and confusion matrices, all on a stratified 80/20 validation split.
5. **Kaggle submission**: predict on the test set with Logistic Regression
   ([`submissions/`](submissions)).

## Results (validation accuracy)

| Model | Task 1: hand-crafted | Task 1: ResNet50 | Task 2: hand-crafted | Task 2: ResNet50 |
|-------|:-------------------:|:---------------:|:-------------------:|:---------------:|
| KNN                 | 0.263 | 0.745 | 0.131 | 0.607 |
| Logistic Regression | 0.545 | **0.904** | 0.250 | 0.845 |
| SVM (RBF)           | 0.520 | 0.869 | 0.333 | 0.798 |
| Linear SVM          | –     | 0.875 | –     | 0.845 |
| MLP (PyTorch)       | –     | 0.893 | –     | **0.857** |

Pretrained ResNet50 features beat the hand-crafted features by a wide margin on both tasks. That gap
matters most on the small, fine-grained bird dataset, where hand-crafted features barely beat chance.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook image_classification_pipeline.ipynb
```

The datasets aren't included in this repo. The notebook expects this layout:

```
task1_data/
  color_histogram.csv  hog_pca.csv  additional_features.csv
  images/train/<class>/*.png   images/test/*.png
task2_data/   (same structure)
train_metadata.csv
train_metadata2.csv
```

ResNet50 feature extraction is slow on a CPU. With a CUDA GPU the notebook uses it automatically.
