# Food-101 Image Classification: CNN from Scratch vs. Transfer Learning

CECS 456 Machine Learning project. We classify food photos from a 20-class subset of
[Food-101](https://www.kaggle.com/datasets/dansbecker/food-101) and compare two models:

- **CNN trained from scratch** (`1_scratch_cnn.ipynb`), by Member 1 (name)
- **ResNet-50 transfer learning** (`2_transfer_resnet50.ipynb`), by Member 2 (name)

Both models use the same 20 classes, the same train/validation/test split
(600 / 150 / 250 images per class) and the same metrics, defined in `common.py`.

## How to run

1. Install the packages: `pip install -r requirements.txt` (or open the notebooks in Google Colab with a GPU).
2. Run `1_scratch_cnn.ipynb` and `2_transfer_resnet50.ipynb`. The first run downloads Food-101 (~5 GB) into `data/` and creates `split.json`.
3. Run `3_compare_results.ipynb` to compare the two models.

Results (weights, learning curves, test metrics) are saved in `results/`.
