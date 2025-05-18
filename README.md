# TabNet Credit Risk Analysis

This project demonstrates the application of TabNet, a deep learning architecture, for credit risk modeling using the German Credit dataset within R's tidymodels framework.

## Overview

Credit risk assessment is a critical function in financial institutions. This project explores how TabNet, an attention-based neural network architecture, can be applied to predict credit risk with interpretable results. The implementation leverages the tidymodels ecosystem in R for a structured modeling workflow.

## Dataset

The analysis uses the German Credit dataset from the UCI Machine Learning Repository:
- 1000 credit applications with 20 features
- Binary classification target: good vs. bad credit risk
- Features include: checking account status, credit history, loan purpose, loan amount, savings status, employment duration, and more

## Key Features

- Implementation of TabNet within the tidymodels framework
- Hyperparameter optimization using ANOVA-based racing methods
- Variable importance visualization to enhance model interpretability
- Cross-validation for robust performance evaluation

## Technical Stack

- **R**: Primary programming language
- **tidyverse**: For data manipulation and visualization
- **tidymodels**: For modeling pipeline and workflow
- **torch**: Deep learning backend
- **tabnet**: R implementation of TabNet architecture
- **finetune**: For hyperparameter optimization
- **vip**: For variable importance visualization

## Model Performance

The model is evaluated using multiple metrics:
- Accuracy
- Precision
- Recall
- ROC AUC

## Getting Started

### Prerequisites

This project requires R with the following packages:
```r
packages <- c("tidyverse", "torch", "tabnet", "tidymodels", "finetune", "vip")
```

### Installation

```r
# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}
```

### Running the Analysis

Open the RMarkdown file `TidyTabNetCredit.Rmd` in RStudio and run all chunks to reproduce the analysis.

## Model Building Process

1. **Data Preprocessing**: Converting categorical variables to factors and normalizing numeric features
2. **Initial Model**: Training TabNet with default parameters from the original paper
3. **Hyperparameter Tuning**: Using grid search with ANOVA-based racing method
4. **Final Model**: Building optimized model with tuned parameters
5. **Evaluation**: Assessing model performance on holdout test data

## Benefits of TabNet

- **Feature Selection**: Built-in mechanisms to highlight important features
- **Interpretability**: Provides feature attribution at each decision step
- **Performance**: Competitive with other state-of-the-art tabular data models

## License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

## References

- Original TabNet paper: [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
- German Credit Dataset: [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/)
- TabNet R Package: [mlverse/tabnet](https://github.com/mlverse/tabnet)