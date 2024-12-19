# Predicting the Manner of Exercise with Accelerometer Data

## Project Overview

This repository contains the code and report for the "Practical Machine Learning" course project on Coursera. The project focuses on predicting the manner in which participants performed a weightlifting exercise (the "classe" variable) using data collected from accelerometers on the belt, forearm, arm, and dumbbell. The dataset used in this project is the [Weight Lifting Exercise Dataset](http://groupware.les.inf.puc-rio.br/har) from the UCI Machine Learning Repository.

The analysis includes data exploration, preprocessing, feature selection using ANOVA, dimensionality reduction using PCA, and model selection using cross-validation. The final model chosen is a Random Forest trained on the full preprocessed dataset.

## Repository Structure

*   `Report.Rmd`: The R Markdown file containing the full project report, including code, explanations, and visualizations.
*   `Report.html`: The compiled HTML version of the report, hosted on GitHub Pages.
*   `data/`: (Optional) A folder containing the `pml-training.csv` and `pml-testing.csv` data files.
*   `code/`: A folder containing an R script which contains the code used for the analysis.


## Viewing the Report

The compiled HTML report can be viewed online here:

[https://Tescoh.github.io/MLproject-Predicting-exercise-manner/Report.html]


## Reproducing the Analysis

To reproduce the analysis, follow these steps:

1.  Clone this repository to your local machine:

    ```bash
    git clone https://github.com/Tescoh/MLproject-Predicting-exercise-manner.
    ```

2.  Open the `report.Rmd` file in RStudio.

3.  Ensure that you have all the required R packages installed. You can install them using the following code:

    ```R
    install.packages(c("caret", "corrplot", "randomForest", "gbm", "ggplot2")) #add other packages if required
    ```


4.  Run the code chunks in the `report.Rmd` file, making sure to adjust any file paths or settings as needed.

**Note:** The model training process may take some time, especially for the Gradient Boosting Machine (GBM).# MLproject-Predicting-exercise-manner