# Housing Prices Modeling Project
Intermediate Data Science Final Project - Predicting Market Housing Prices

This is the final project for the Intermediate Data Science course at the University of Pittsburgh. The goal was to build a model to predict housing prices in two hypothetical markets using a scraped data set. 10 days were given to complete the project. The deliverables for the project included: 

1. Predictions: A single csv file with 600 test observations.
2. Technical Report: A pdf report that outlines the process from start to finish in technical detail.
3. Final (non-technical) Report: A pdf report that discusses the findings to a non-technical decision maker.
4. Code: A single .R file with the code.

The intended strategy for the project was to test all the applicable models that we learned in the course to the data to see which model gave the best result. The models tested include:

Model Group                 | Model
----------------------------|-----------------------------------------------------------
Linear Models               | Forward, Backward, Best Subset Selection
Shrinkage & Regularization  | Lasso, Ridge Regression, KNN, PCR, PLS
Nonlinear Models            | Polynomial, Splines, GAMs
Trees & Ensembles           | Regression Tree, Random Forest (RF), Bagging RF, Boosted RF

## Project Highlights
* There were three challenges that made the project more difficult, handling missing values, removing outliers, and troubleshooting errors with a large categorical variable. 
* The model’s predictive accuracy is measured using mean squared error. Based on MSE, the Trees and Ensembles Group perform the best, with the random forest model performing the best overall.

The project received an A+ grade overall. 
