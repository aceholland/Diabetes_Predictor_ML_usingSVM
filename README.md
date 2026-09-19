# Diabetes_Predictor_ML_usingSVM
A Machine Learning classification project that predicts whether a person is diabetic based on medical diagnostic measurements using a **Support Vector Machine (SVM)** classifier.

## Project Overview

This project uses the **Pima Indians Diabetes Dataset** to build a binary classification model for predicting diabetes.

The project follows a complete Machine Learning workflow, including:

- Data collection and analysis
- Exploratory data analysis
- Statistical analysis
- Feature and target separation
- Data standardization
- Train-test splitting
- SVM model training
- Model evaluation
- Prediction on new patient data

## Dataset

The project uses the **Pima Indians Diabetes Dataset**.

The dataset contains **768 patient records** and **8 input features**.

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of times the patient has been pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age of the patient |

### Target

The `Outcome` column is the target variable:

- `0` → Non-Diabetic
- `1` → Diabetic

The dataset contains:

- **500 non-diabetic cases**
- **268 diabetic cases**

## Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Jupyter Notebook**

## Machine Learning Model

### Support Vector Machine (SVM)

A **Support Vector Machine** classifier with a **linear kernel** is used for binary classification.

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
````

Before training the model, the input features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)
```

Standardization brings the features onto a comparable scale, which is particularly important for SVM models.

## Project Workflow

### 1. Import Dependencies

The project uses NumPy, Pandas, and Scikit-learn for data processing, standardization, model training, and evaluation.

### 2. Load the Dataset

The dataset is loaded using Pandas:

```python
diabetes_dataset = pd.read_csv('diabetes.csv')
```

### 3. Explore the Dataset

The first few records are examined using:

```python
diabetes_dataset.head()
```

The dimensions of the dataset are checked using:

```python
diabetes_dataset.shape
```

The statistical characteristics of the dataset are analyzed using:

```python
diabetes_dataset.describe()
```

The dataset contains:

* **768 rows**
* **9 columns**

### 4. Analyze the Target Variable

The distribution of diabetic and non-diabetic cases is examined using:

```python
diabetes_dataset['Outcome'].value_counts()
```

The mean values of the features for each outcome are also calculated:

```python
diabetes_dataset.groupby('Outcome').mean()
```

This provides an initial understanding of how the input features differ between the two outcome classes.

### 5. Separate Features and Target

The `Outcome` column is separated from the input features:

```python
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
```

Here:

* `X` contains the input features
* `Y` contains the target labels

### 6. Standardize the Data

The input features are standardized using `StandardScaler`:

```python
scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)
```

The standardized data is then used for model training and testing.

### 7. Train-Test Split

The dataset is divided into training and testing sets:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)
```

This creates:

* **80% training data**
* **20% testing data**

The `stratify=Y` parameter helps maintain the distribution of the target classes across the training and testing sets.

### 8. Train the SVM Model

A Support Vector Machine classifier using a linear kernel is created:

```python
classifier = svm.SVC(kernel='linear')
```

The model is then trained:

```python
classifier.fit(X_train, Y_train)
```

## Model Evaluation

The trained model is evaluated using **classification accuracy**.

### Training Accuracy

**78.66%**

### Testing Accuracy

**77.27%**

| Dataset  |   Accuracy |
| -------- | ---------: |
| Training | **78.66%** |
| Testing  | **77.27%** |

The training and testing accuracies are relatively close, indicating that there is not a large performance gap between the training and testing data in this experiment.

## Making Predictions

The trained model can be used to predict the outcome for a new patient.

Example input:

```python
input_data = (4, 110, 92, 0, 37.6, 0, 0.191, 30)
```

The input data is converted into a NumPy array:

```python
input_data_as_numpy_array = np.asarray(input_data)
```

The array is reshaped because the model expects the input in a two-dimensional format:

```python
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
```

The input is then standardized using the same scaler used during preprocessing:

```python
std_data = scaler.transform(input_data_reshaped)
```

Finally, the standardized input is passed to the trained SVM model:

```python
prediction = classifier.predict(std_data)
```

The prediction is converted into a readable result:

```python
if prediction[0] == 0:
    print("Non-Diabetic")
else:
    print("Diabetic")
```

### Example Output

```text
Non-Diabetic
```

## Results

The SVM classifier achieved the following results:

| Metric            |      Score |
| ----------------- | ---------: |
| Training Accuracy | **78.66%** |
| Testing Accuracy  | **77.27%** |

The model provides a baseline for diabetes classification using the selected features and a linear SVM classifier.

> Accuracy alone does not fully describe performance for a medical classification problem. Metrics such as precision, recall, F1-score, confusion matrix, and ROC-AUC can provide additional insight.

## Project Structure

```text
diabetes-prediction-svm/
│
├── diabetes.csv
├── diabetes_prediction.ipynb
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/diabetes-prediction-svm.git
```

### 2. Navigate to the Project Directory

```bash
cd diabetes-prediction-svm
```

### 3. Install Dependencies

```bash
pip install numpy pandas scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook and run the cells sequentially.

## Key Concepts Demonstrated

This project demonstrates several fundamental Machine Learning concepts:

* **Binary Classification**
* **Support Vector Machines**
* **Linear Kernel**
* **Feature Standardization**
* **StandardScaler**
* **Train-Test Split**
* **Stratified Sampling**
* **Exploratory Data Analysis**
* **Statistical Analysis**
* **Model Training**
* **Model Evaluation**
* **Prediction on New Data**

## Future Improvements

The project can be extended in several ways:

* Add a confusion matrix
* Calculate precision, recall, and F1-score
* Evaluate ROC-AUC
* Compare SVM with Logistic Regression
* Compare with Random Forest and XGBoost
* Perform hyperparameter tuning
* Perform cross-validation
* Improve feature preprocessing
* Investigate zero values that may represent missing measurements
* Build an interactive web application using Streamlit
* Deploy the model as a web application

## Disclaimer

This project is created for **educational and Machine Learning demonstration purposes only**.

It is **not a medical diagnostic system** and should not be used to make clinical decisions.

## Author

**Anushka Verma**

GitHub: https://github.com/aceholland/Diabetes_Predictor_ML_usingSVM
