# Classification, Regression & Clustering

An end-to-end Machine Learning project that brings together three major ML approaches—**Classification, Regression, and Clustering**—in one interactive application built using Python and Streamlit.

The project demonstrates how machine learning can solve different types of real-world problems, from identifying electrical appliances to predicting Airbnb prices and segmenting customers.

## Project Overview

This project integrates three independent machine learning pipelines into a unified application.

| Module | ML Approach | Objective |
|---|---|---|
| Classification | Supervised learning | Identify electrical appliances from their electrical measurements |
| Regression | Supervised learning | Predict Airbnb nightly prices |
| Clustering | Unsupervised learning | Group customers into meaningful segments |

Each module uses a trained model and its corresponding preprocessing steps to generate predictions or identify patterns from user-provided inputs.

## Project Objectives

- Implement classification, regression, and clustering algorithms.
- Perform data preprocessing and feature engineering.
- Train and evaluate machine learning models.
- Compare candidate algorithms and select suitable final models.
- Save trained models and preprocessing components for reuse.
- Build an interactive Streamlit application.
- Deploy the application so users can interact with the models through a web interface.

## 1. Classification — Smart Appliance Identifier

### Objective

Predict the type of electrical appliance from its electrical measurements.

### Dataset and features

The classification pipeline uses electrical appliance data. Engineered features include power, apparent power, power factor, and load share.

The deployed model uses seven electrical input features. The exact seven feature names should match the final model's saved feature configuration.

### Algorithms explored

- Random Forest
- AdaBoost
- Bagging Classifier
- Multi-Layer Perceptron (MLP)
- Gradient Boosting
- Support Vector Machine (SVM)
- Other candidate algorithms included in the model comparison experiments

### Model development

The classification experiments include hyperparameter tuning, comparison of multiple algorithms, and five-fold cross-validation.

The final deployed classifier is a **Random Forest Classifier** with:

- Number of trees: 25
- Minimum samples per leaf: 10

### Reported results

| Evaluation metric | Result |
|---|---:|
| Accuracy | Approximately 95.3% |
| F1-score | Approximately 0.953 |
| ROC-AUC | Approximately 0.998 |

The remaining classification errors are concentrated in distinguishing Iron and Electric Kettle observations, suggesting that these appliances have similar electrical characteristics in the dataset.

### Application input and output

**Input:** Seven electrical measurement features.

**Output:** Predicted appliance category.

## 2. Regression — Airbnb Price Prediction

### Objective

Estimate the nightly price of an Airbnb listing based on its listing details.

### Algorithm

The final regression model is a **Random Forest Regressor**.

### Workflow

1. Load the Airbnb dataset.
2. Inspect and preprocess the listing features.
3. Prepare the input variables and target price.
4. Train and compare regression models.
5. Select the final Random Forest Regressor.
6. Save the trained model and required preprocessing components.
7. Use the saved model to predict prices for new listings.

### Evaluation metrics

Regression performance should be reported using the metrics calculated in the regression notebook:

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted prices.
- **MSE (Mean Squared Error):** Average squared prediction error.
- **RMSE (Root Mean Squared Error):** Square root of MSE, expressed in the target's units.
- **R² Score:** Measures how much of the variation in the target is explained by the model.

**Final model:** Random Forest Regressor (`rf`).

**Application input:** Airbnb listing details.

**Output:** Estimated nightly price.

*Note: Actual regression scores, the dataset source, and the exact input features should be added from the completed regression notebook.*

## 3. Clustering — Customer Segmentation

### Objective

Group customers into clusters based on similarities in their selected characteristics.

Unlike classification, clustering does not require predefined class labels. The algorithm discovers groups from the data.

### Algorithm

The final clustering model is **K-Means** (`kmeans_final`).

### Workflow

1. Load and inspect the customer dataset.
2. Select relevant customer features.
3. Handle missing values and prepare the data.
4. Scale the selected features using the fitted scaler.
5. Train and evaluate K-Means clustering.
6. Save the fitted clustering model and scaler.
7. Assign new customer inputs to the learned clusters.

### Evaluation metrics

Depending on the completed experiments, clustering can be assessed using:

- **Silhouette Score:** Measures how well-separated and cohesive the clusters are.
- **Inertia:** Measures the within-cluster sum of squared distances.
- **Davies–Bouldin Index:** Measures cluster similarity; lower values generally indicate better separation.

These metrics should only be reported when they have actually been calculated.

**Final model:** K-Means (`kmeans_final`).

**Application input:** Customer information matching the training features.

**Output:** Assigned cluster, with an interpretation of the segment where available.

*Note: The dataset source, final number of clusters, feature names, and actual evaluation scores should be added from the completed clustering notebook.*

## Technologies Used

- **Programming language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Model persistence:** Joblib
- **Web application:** Streamlit
- **Development environment:** Jupyter Notebook
- **Version control:** Git and GitHub

## Project Structure

```text
Unified-ML-Studio/
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── models/
│   ├── final_model.joblib
│   ├── regression_bundle.joblib
│   └── clustering_bundle.joblib
│
├── notebooks/
│   ├── classification.ipynb
│   ├── regression.ipynb
│   └── clustering_completed.ipynb
│
├── scripts/
│   ├── export_regression.py
│   └── export_clustering.py
│
├── README.md
└── .gitignore
```

*The structure above is the intended organization. Adjust notebook and script filenames to match the files actually committed to the repository.*

## Model Deployment

The project uses Streamlit to provide a common interface for all three machine learning modules.

The application allows users to select a task, enter the required inputs, and view the corresponding result.

- **Classification:** Appliance identification.
- **Regression:** Airbnb nightly-price estimation.
- **Clustering:** Customer segment assignment.

The saved model bundles must include all preprocessing components needed at inference time. For example, clustering must use the scaler fitted during training rather than fitting a new scaler to each prediction.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version for the installed dependencies.
- Git.
- The project repository and saved model files.

### Step 1: Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Unified-ML-Studio
```

Replace the placeholder with the actual GitHub repository URL.

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r app/requirements.txt
```

Ensure the requirements file includes all libraries imported by the application.

### Step 4: Verify the model files

Confirm that the saved model files exist in the `models/` directory and that the paths used in `app/app.py` match their actual locations.

## Running the Application

From the repository root, run:

```bash
streamlit run app/app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to interact with the application.

## Model Evaluation

The project evaluates models using metrics appropriate to each ML task.

| Task | Main evaluation measures |
|---|---|
| Classification | Accuracy, precision, recall, F1-score, ROC-AUC |
| Regression | MAE, MSE, RMSE, R² |
| Clustering | Silhouette Score, inertia, Davies–Bouldin Index |

The classification results reported above are from the model experiments. Regression and clustering results should be populated from their respective evaluation outputs.

## Future Improvements

- Add visual comparisons of model performance.
- Display evaluation metrics and relevant plots in the application.
- Improve model interpretability using feature importance and other explanation techniques.
- Add stronger input validation and error handling.
- Expand the datasets and evaluate generalization on unseen samples.
- Automate model retraining and versioning.
- Improve deployment monitoring and documentation.

## Conclusion

Unified ML Studio demonstrates the use of three complementary machine learning approaches in a single application. Classification predicts discrete categories, regression estimates continuous values, and clustering discovers groups in unlabeled data.

By combining model development, evaluation, model persistence, and interactive deployment, the project provides an end-to-end demonstration of practical machine learning workflows.

---

**Project status:** Classification model results documented; regression and clustering evaluation details to be completed from their respective notebooks.
