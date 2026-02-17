# SVM Image Classification (Outdoor-Day vs Other)

This project builds an image classifier to predict whether a photo is **outdoor-day** using:
- Baseline RGB median features (3 features)
- Grid-based RGB medians (10×10 grid → 300 features)
- PCA for dimensionality reduction (keeps components explaining ≥95% variance)
- SVM (radial kernel) for classification

## Files
- `svm-image-classification.r` — main script
- `model_comparison_results.csv` — saved model comparison output (generated)

## Data Setup (Required)
This script expects:
- `data/photoMetaData.csv`
- `data/columbiaImages/` (folder containing the JPEG images referenced in the metadata)

**Important:** The repository should use relative paths, not `/Users/...` paths.

## R Packages
Install once:
```r
install.packages(c("jpeg", "e1071"))
