# Image Classification Pipeline

A pipeline that compares hand-crafted image features against deep features from a pretrained ResNet50, using a handful of classifiers on two datasets. Everything is in `image_classification_pipeline.ipynb`.

## Datasets

- **Task 1:** 10 animal classes (bird, butterfly, cat, deer, dog, elephant, frog, horse, sheep, spider). 3,750 training images and 1,250 test images.
- **Task 2:** 10 fine-grained bird species (Blue Jay, Cardinal, Herring Gull, Song Sparrow, etc.). Only 417 training images and 180 test images, which makes it a lot harder.

Each dataset includes the raw images and three precomputed feature sets: colour histograms, HOG reduced with PCA, and some additional features.

## How it works

I started by merging the three precomputed feature sets, standardising them with `StandardScaler`, and training KNN, Logistic Regression and an RBF SVM as a baseline. These features didn't get very far: about 55% on Task 1 and 33% on Task 2.

Next I passed every image through an ImageNet-pretrained ResNet50 with the final fully connected layer replaced by an identity. Each image was resized to 224×224 and normalised with ImageNet stats. This gives a 2048-dimensional embedding per image, and the notebook uses a GPU when one is available.

On the embeddings I trained KNN, Logistic Regression, an RBF SVM (tuned with `GridSearchCV` over C and gamma, which picked C=10 with gamma='scale'), a Linear SVM, and a small MLP written in PyTorch. For Task 2 I also tried concatenating the ResNet features with the hand-crafted ones. It came out slightly worse, so I kept ResNet only.

Every model is evaluated on a stratified 80/20 validation split. I report accuracy, balanced accuracy, macro precision, macro recall (sensitivity), macro specificity, and macro and weighted F1, and I plot a confusion matrix for each model.

## Results

The ResNet features made a huge difference. On Task 1, Logistic Regression went from 55% to 90% validation accuracy, with the MLP at 89%, Linear SVM at 87.5%, RBF SVM at 87% and KNN at 75%. On Task 2 the MLP did best at 86%, with Logistic Regression and Linear SVM at 84.5%, RBF SVM at 80% and KNN at 61%.

I used Logistic Regression for the final test set predictions, which are saved in `submissions/`.

## Running it

```
pip install -r requirements.txt
jupyter notebook image_classification_pipeline.ipynb
```

The data is already in `task1_data/` and `task2_data/`. ResNet feature extraction is slow on a CPU, so give it a while if you don't have a GPU. The extracted features are saved to `resnet_train.csv` and `resnet_test.csv` in each data folder.
