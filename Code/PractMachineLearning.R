# Load libraries
library(caret)
library(ggplot2)
library(corrplot)
library(randomForest)
# Load data
trainData <- read.csv("pml-training.csv")
testData <- read.csv("pml-testing.csv")

# Basic exploration of the training data
summary(trainData)
str(trainData)
sum(is.na(trainData)) #count total NAs in training data
head(trainData)
table(trainData$classe)

# Basic exploration of the testing data
summary(testData)
str(testData)
sum(is.na(testData)) #count total NAs in training data
head(testData)

# 1. Handle Missing Values: Remove columns with many NAs

# Calculate the number of NAs in each column
na_counts <- colSums(is.na(trainData))

# Define a threshold for missing values (e.g., 95%)
threshold <- 0.95 * nrow(trainData)

# Select columns where the number of NAs is less than the threshold
trainData1 <- trainData[, na_counts < threshold]

# 2. Remove Irrelevant Variables

# Define a vector of irrelevant column names
irrelevant_cols <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")

# Remove these columns from the training data
trainData2 <- trainData1[, !names(trainData1) %in% irrelevant_cols]

# 3. Remove variables with near-zero variance

# Identify near-zero variance predictors
nzv_cols <- nearZeroVar(trainData2)

# Remove these columns from the training data
trainData3 <- trainData2[,-nzv_cols]

# 4. Remove rows with NA's

# Remove rows with NAs
trainData4 <- na.omit(trainData3)

# Check that no more NA's exist in the data
sum(is.na(trainData4))

# Convert classe to a factor (if not already)
trainData4$classe <- as.factor(trainData4$classe)

# Check the updated structure and summary
str(trainData4)
summary(trainData4)

# --- Univariate Feature Selection using ANOVA ---

numeric_data <- trainData4[, sapply(trainData4, is.numeric)]

# Store the p-values
p_values <- numeric(ncol(numeric_data))
names(p_values) <- names(numeric_data)

# Perform ANOVA for each numeric variable
for (i in 1:ncol(numeric_data)) {
  # Create a linear model with the current variable as the predictor and 'classe' as the outcome
  model <- lm(numeric_data[, i] ~ trainData4$classe)
  # Perform ANOVA on the model
  anova_result <- anova(model)
  # Store the p-value from the ANOVA table
  p_values[i] <- anova_result$"Pr(>F)"[1]
}

# Sort variables by p-value (ascending order - smaller p-value means more significant)
sorted_p_values <- sort(p_values)

# Select top N variables based on p-values (e.g., top 20)
N <- 20
selected_variables_anova <- names(sorted_p_values)[1:N]

# --- Option 1: Selective Histograms and Boxplots ---
# Histograms and box plots for variables selected using ANOVA
for (var in selected_variables_anova) {
  hist(trainData4[, var], 
       main = paste("Histogram of", var),
       xlab = var,
       col = "lightblue",
       border = "black")
}

# Boxplots for selected variables vs. classe
# Boxplots for selected variables vs. classe
for (var in selected_variables_anova) {
  boxplot(trainData4[, var] ~ trainData4$classe,
          main = paste("Boxplot of", var, "vs. Classe"),
          xlab = "Classe",
          ylab = var,
          col = "lightblue",
          border = "black")
}

# Perform PCA (scale. = TRUE is important for variables on different scales)
pca_result <- prcomp(numeric_data, scale. = TRUE)

# Scree plot to visualize explained variance by each principal component
plot(pca_result, type = "l", main = "Scree Plot")

# Get the principal component scores
pca_scores <- as.data.frame(pca_result$x)

# Add the 'classe' variable to the PCA scores for visualization
pca_scores$classe <- trainData4$classe

# Scatter plot of the first two principal components, colored by classe
plot(pca_scores$PC1, pca_scores$PC2, 
     col = trainData4$classe, pch = 19,
     xlab = "PC1", ylab = "PC2",
     main = "PCA: First Two Principal Components")
legend("topright", legend = levels(trainData4$classe), col = 1:5, pch = 19)

# Boxplots of the first principal component vs. classe
ggplot(pca_scores, aes(x = classe, y = PC1)) +
  geom_boxplot(fill = "lightblue", colour = "black") +
  ggtitle("Boxplot of PC1 vs. Classe") +
  xlab("Classe") +
  ylab("PC1") +
  theme_minimal()

#2. Correlation Matrix using corrplot

# Calculate the correlation matrix for numeric data
cor_matrix <- cor(numeric_data)

# Visualize the correlation matrix using corrplot
corrplot(cor_matrix, 
         method = "color",           # Use colored circles to represent correlations
         type = "upper",             # Display the upper triangle of the matrix
         order = "hclust",           # Order variables using hierarchical clustering
         addCoef.col = "black",     # Add correlation coefficients in black
         tl.col = "black",           # Color of the text labels (variable names)
         tl.srt = 45,                # Rotate text labels by 45 degrees
         diag = FALSE,               # Do not display the main diagonal (self-correlations)
         title = "Correlation Matrix of Numeric Variables", #title
         mar=c(0,0,2,0)) #adjust margins so title displays properly


# --- Step 5: Model Training and Cross-Validation ---

# ANOVA Dataset
trainData_anova <- trainData4[, c(selected_variables_anova, "classe")]

# PCA Dataset (using first 20 PCs)

trainData_pca <- pca_scores[, c(paste0("PC", 1:20), "classe")]

# 2. Set up trainControl

# Define cross-validation method (10-fold CV)
train_control <- trainControl(method = "cv", number = 10)

# 3. Train and Tune Models

# --- Random Forest (rf) ---

# Define tuning grid for mtry (number of variables randomly sampled at each split)
rf_tuneGrid <- expand.grid(mtry = c(2, 3, 5, 7, 9))

# Train RF on Full Dataset
set.seed(123) # For reproducibility
rf_full <- train(classe ~ ., data = trainData4, method = "rf", trControl = train_control, tuneGrid = rf_tuneGrid)

# Train RF on ANOVA Dataset
set.seed(123)
rf_anova <- train(classe ~ ., data = trainData_anova, method = "rf", trControl = train_control, tuneGrid = rf_tuneGrid)

# Train RF on PCA Dataset
set.seed(123)
rf_pca <- train(classe ~ ., data = trainData_pca, method = "rf", trControl = train_control, tuneGrid = rf_tuneGrid)

# --- Gradient Boosting Machine (gbm) ---

# Define tuning grid for GBM parameters
gbm_tuneGrid <- expand.grid(
  n.trees = c(100, 200, 300),           # Number of trees
  interaction.depth = c(1, 2, 3),     # Tree depth
  shrinkage = c(0.01, 0.1),             # Learning rate
  n.minobsinnode = c(5, 10)           # Minimum observations in terminal nodes
)

# Train GBM on Full Dataset
set.seed(123)
gbm_full <- train(classe ~ ., data = trainData4, method = "gbm", trControl = train_control, tuneGrid = gbm_tuneGrid, verbose = FALSE)

# Train GBM on ANOVA Dataset
set.seed(123)
gbm_anova <- train(classe ~ ., data = trainData_anova, method = "gbm", trControl = train_control, tuneGrid = gbm_tuneGrid, verbose = FALSE)

# Train GBM on PCA Dataset
set.seed(123)
gbm_pca <- train(classe ~ ., data = trainData_pca, method = "gbm", trControl = train_control, tuneGrid = gbm_tuneGrid, verbose = FALSE)

# 4. Evaluate Performance

# Create a function to extract performance metrics
get_performance <- function(model) {
  best_tune <- model$bestTune
  performance <- model$results[which.max(model$results$Accuracy), ]
  return(performance)
}

# Get performance for each model and dataset
rf_full_performance <- get_performance(rf_full)
rf_anova_performance <- get_performance(rf_anova)
rf_pca_performance <- get_performance(rf_pca)

gbm_full_performance <- get_performance(gbm_full)
gbm_anova_performance <- get_performance(gbm_anova)
gbm_pca_performance <- get_performance(gbm_pca)

# 5. Compare Performance

# Create a data frame to store the results
performance_summary <- data.frame(
  Model = c("Random Forest", "Random Forest", "Random Forest", "GBM", "GBM", "GBM"),
  Dataset = c("Full", "ANOVA", "PCA", "Full", "ANOVA", "PCA"),
  Accuracy = c(rf_full_performance$Accuracy, rf_anova_performance$Accuracy, rf_pca_performance$Accuracy,
               gbm_full_performance$Accuracy, gbm_anova_performance$Accuracy, gbm_pca_performance$Accuracy),
  Kappa = c(rf_full_performance$Kappa, rf_anova_performance$Kappa, rf_pca_performance$Kappa,
            gbm_full_performance$Kappa, gbm_anova_performance$Kappa, gbm_pca_performance$Kappa)
)

# Print the performance summary
print(performance_summary)

# Visualize performance comparison
comparison_plot <- ggplot(performance_summary, aes(x = Dataset, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Model Performance Comparison") +
  theme_minimal()
print(comparison_plot)

# 1. Train Final Random Forest Model on the Entire Training Dataset

# Get the best mtry value from the cross-validation results of rf_full
best_mtry <- rf_full$bestTune$mtry

# Train the final Random Forest model on the entire training set
set.seed(123)  # For reproducibility
final_rf_model <- randomForest(classe ~ ., data = trainData4, mtry = best_mtry)

# 2. Preprocess the Test Set

# Apply the same preprocessing steps to testData as we did to trainData


# Remove columns with many NAs (using the same threshold as for the training data)
testData1 <- testData[, na_counts < threshold]

# Remove irrelevant columns
testData2 <- testData1[, !names(testData1) %in% irrelevant_cols]

# Remove near-zero variance columns (using the same columns as identified in the training data)
testData3 <- testData2[, -nzv_cols]

# 3. Make Predictions on the Test Set

# Make predictions using the final Random Forest model
final_predictions <- predict(final_rf_model, newdata = testData)

# 4. Format Predictions

# Print the predictions
print(final_predictions)

save.image("practmachinelearning.RData")