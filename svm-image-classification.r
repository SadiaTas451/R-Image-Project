# Install required packages if needed (run once)
# install.packages(c("jpeg", "e1071"))

# Load required libraries
library(jpeg)    # For reading JPEG images
library(e1071)   # For SVM

#############################################
# 1. DATA PREPARATION
#############################################

# Read metadata
photo_metadata <- read.csv("/Users/sadiatasnim/Downloads/Project-20250411/photoMetaData.csv")

# Create binary target variable (1 if outdoor-day, 0 otherwise)
photo_metadata$y <- as.numeric(photo_metadata$category == "outdoor-day")

# Create train/test split (same as sample code)
set.seed(123)
n <- nrow(photo_metadata)
train_flag <- (runif(n) > 0.5)

#############################################
# 2. FEATURE EXTRACTION FUNCTIONS
#############################################

# Function to properly handle file paths
get_image_path <- function(filename) {
  return(paste0("/Users/sadiatasnim/Downloads/Project-20250411/columbiaImages/", filename))
}

# Grid-based feature extraction with 10x10 grid
extract_grid_features <- function(img_path, grid_size = 10) {
  # Read image
  img <- readJPEG(img_path)
  
  # Get image dimensions
  height <- dim(img)[1]
  width <- dim(img)[2]
  
  # Calculate grid cell dimensions
  cell_height <- height %/% grid_size
  cell_width <- width %/% grid_size
  
  # Initialize feature vector
  features <- numeric(grid_size * grid_size * 3)
  
  # Extract features from each grid cell
  feature_idx <- 1
  for (i in 1:grid_size) {
    # Calculate row bounds for this grid cell
    row_start <- (i-1) * cell_height + 1
    row_end <- min(i * cell_height, height)
    
    for (j in 1:grid_size) {
      # Calculate column bounds for this grid cell
      col_start <- (j-1) * cell_width + 1
      col_end <- min(j * cell_width, width)
      
      # Extract grid cell
      cell <- img[row_start:row_end, col_start:col_end, ]
      
      # Calculate median RGB values for this cell
      for (channel in 1:3) {
        features[feature_idx] <- median(cell[,,channel])
        feature_idx <- feature_idx + 1
      }
    }
  }
  
  return(features)
}

# Extract baseline features (3 RGB medians for whole image)
extract_baseline_features <- function(img_path) {
  img <- readJPEG(img_path)
  return(apply(img, 3, median))
}

# Custom function to calculate ROC curve and AUC 
calculate_roc <- function(actual, predicted) {
  # Sort by predicted probability
  ord <- order(predicted, decreasing = TRUE)
  actual <- actual[ord]
  predicted <- predicted[ord]
  
  # Calculate TPR and FPR at different thresholds
  n_pos <- sum(actual)
  n_neg <- length(actual) - n_pos
  
  tpr <- cumsum(actual) / n_pos
  fpr <- cumsum(!actual) / n_neg
  
  # Include (0,0) and (1,1) points
  tpr <- c(0, tpr, 1)
  fpr <- c(0, fpr, 1)
  
  # Calculate AUC using trapezoidal rule
  auc_value <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
  
  return(list(fpr = fpr, tpr = tpr, auc = auc_value))
}

#############################################
# 3. MAIN ANALYSIS
#############################################

run_analysis <- function() {
  # 1. Extract baseline features (for comparison)
  print("Extracting baseline features...")
  X_baseline <- matrix(NA, ncol=3, nrow=n)
  for (j in 1:n) {
    img_path <- get_image_path(photo_metadata$name[j])
    X_baseline[j,] <- extract_baseline_features(img_path)
    if (j %% 100 == 0) print(sprintf("Baseline: %03d / %03d", j, n))
  }
  colnames(X_baseline) <- c("R_median", "G_median", "B_median")
  
  # 2. Extract grid features
  print("Extracting grid features...")
  grid_size <- 10
  X_grid <- matrix(NA, ncol = grid_size * grid_size * 3, nrow = n)
  for (j in 1:n) {
    img_path <- get_image_path(photo_metadata$name[j])
    X_grid[j,] <- extract_grid_features(img_path, grid_size)
    if (j %% 50 == 0) print(sprintf("Grid features: %03d / %03d", j, n))
  }
  
  # Create column names for grid features
  col_names <- character(grid_size * grid_size * 3)
  idx <- 1
  for (i in 1:grid_size) {
    for (j in 1:grid_size) {
      for (c in c("R", "G", "B")) {
        col_names[idx] <- sprintf("Grid_%d_%d_%s", i, j, c)
        idx <- idx + 1
      }
    }
  }
  colnames(X_grid) <- col_names
  
  # 3. Split data into training and testing sets
  X_baseline_train <- X_baseline[train_flag,]
  X_baseline_test <- X_baseline[!train_flag,]
  X_grid_train <- X_grid[train_flag,]
  X_grid_test <- X_grid[!train_flag,]
  y_train <- photo_metadata$y[train_flag]
  y_test <- photo_metadata$y[!train_flag]
  
  # 4. Scale features (important for SVM)
  X_grid_train_scaled <- scale(X_grid_train)
  X_grid_test_scaled <- scale(X_grid_test, 
                             center = attr(X_grid_train_scaled, "scaled:center"),
                             scale = attr(X_grid_train_scaled, "scaled:scale"))
  
  # 5. Apply PCA for dimensionality reduction
  print("Performing PCA on image features...")
  pca_result <- prcomp(X_grid_train_scaled, center = FALSE, scale. = FALSE)
  
  # Calculate variance explained by each principal component
  variance_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
  cumulative_variance <- cumsum(variance_explained)
  
  # Print variance explained information
  print("Variance explained by principal components:")
  for (i in 1:10) {
    print(sprintf("PC%d: %.2f%% (Cumulative: %.2f%%)", 
                  i, variance_explained[i] * 100, cumulative_variance[i] * 100))
  }
  
  # Plot scree plot to visualize variance explained
  barplot(variance_explained[1:20] * 100, 
          main = "Variance Explained by Principal Components",
          xlab = "Principal Component", 
          ylab = "Percent of Variance Explained",
          names.arg = 1:20,
          ylim = c(0, max(variance_explained[1:20] * 100) * 1.2))
  
  # Choose number of principal components to keep
  n_components <- which(cumulative_variance >= 0.95)[1]
  print(paste("Using", n_components, "principal components that explain ≥95% of variance"))
  
  # Project data onto principal components
  X_grid_train_pca <- predict(pca_result, X_grid_train_scaled)[, 1:n_components]
  X_grid_test_pca <- predict(pca_result, X_grid_test_scaled)[, 1:n_components]
  
  #############################################
  # 6. MODEL TRAINING AND EVALUATION
  #############################################
  
  # Function to evaluate a model
  evaluate_model <- function(pred_prob, actual, model_name) {
    pred_class <- ifelse(pred_prob > 0.5, 1, 0)
    accuracy <- mean(pred_class == actual)
    confusion <- table(Predicted = factor(pred_class, levels = c(0, 1)), 
                      Actual = factor(actual, levels = c(0, 1)))
    sensitivity <- confusion[2,2] / sum(confusion[,2])
    specificity <- confusion[1,1] / sum(confusion[,1])
    
    # Calculate ROC curve and AUC
    roc_result <- calculate_roc(actual, pred_prob)
    
    # Print results
    cat("\n", model_name, "Results:\n")
    cat("Accuracy:", accuracy, "\n")
    cat("Sensitivity:", sensitivity, "\n")
    cat("Specificity:", specificity, "\n")
    cat("AUC:", roc_result$auc, "\n")
    print(confusion)
    
    return(list(
      model_name = model_name,
      accuracy = accuracy,
      sensitivity = sensitivity, 
      specificity = specificity,
      auc = roc_result$auc,
      confusion = confusion,
      roc = roc_result
    ))
  }
  
  # Baseline model (logistic regression with just 3 RGB medians)
  print("Training baseline logistic regression model...")
  baseline_model <- glm(y_train ~ ., family = binomial(link = "logit"), 
                       data = data.frame(X_baseline_train))
  baseline_pred_prob <- predict(baseline_model, 
                               newdata = data.frame(X_baseline_test), 
                               type = "response")
  baseline_results <- evaluate_model(baseline_pred_prob, y_test, "Baseline Logistic Regression")
  
  # SVM with PCA features
  print("Training SVM model...")
  # Tune SVM parameters (optional)
  # tune_result <- tune.svm(x = X_grid_train_pca, y = as.factor(y_train),
  #                        gamma = 10^(-5:-1), cost = 10^(0:3),
  #                        tunecontrol = tune.control(cross = 5))
  # print("Best parameters:")
  # print(tune_result$best.parameters)
  
  # Train SVM model with radial kernel
  svm_model <- svm(x = X_grid_train_pca, y = as.factor(y_train), 
                  probability = TRUE, kernel = "radial")
  
  # Get predictions
  svm_pred <- predict(svm_model, X_grid_test_pca, probability = TRUE)
  svm_pred_prob <- attr(svm_pred, "probabilities")[,2]
  
  # Evaluate SVM model
  svm_results <- evaluate_model(svm_pred_prob, y_test, "SVM with PCA")
  
  # Plot ROC curves for comparison
  plot(baseline_results$roc$fpr, baseline_results$roc$tpr, type = "l", col = "black",
       xlab = "False Positive Rate", ylab = "True Positive Rate", 
       main = "ROC Curves Comparison")
  lines(svm_results$roc$fpr, svm_results$roc$tpr, col = "blue")
  abline(0, 1, lty = 2, col = "gray")
  legend("bottomright", 
         legend = c("Baseline (Logistic Regression)", 
                   paste("SVM with PCA (AUC =", round(svm_results$auc, 3), ")")), 
         col = c("black", "blue"), lwd = 2)
  
  # Create comparison table
  comparison <- data.frame(
    Model = c("Baseline Logistic Regression", "SVM with PCA"),
    Accuracy = c(baseline_results$accuracy, svm_results$accuracy),
    Sensitivity = c(baseline_results$sensitivity, svm_results$sensitivity),
    Specificity = c(baseline_results$specificity, svm_results$specificity),
    AUC = c(baseline_results$auc, svm_results$auc)
  )
  
  # Print comparison table
  print("\nModel Comparison:")
  print(comparison)
  
  # Save results for reporting
  write.csv(comparison, "model_comparison_results.csv", row.names = FALSE)
  
  # Return results for further analysis or reporting
  return(list(
    comparison = comparison,
    baseline_model = baseline_model,
    svm_model = svm_model,
    pca_result = pca_result,
    n_components = n_components,
    baseline_results = baseline_results,
    svm_results = svm_results
  ))
}

# Run the analysis
results <- run_analysis()

# Final message
print("\nAnalysis complete. Results saved for reporting.")

