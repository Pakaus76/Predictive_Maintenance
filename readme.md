# Predictive Maintenance System

## Description
This **Predictive Maintenance System** is built using **Streamlit** and a pre-trained **Random Forest** model. It allows users to predict machine failures based on sensor values such as temperature, operational hours, and other key features. The app enables users to adjust input parameters through sliders and receive real-time predictions.

## Dataset
The model has been trained using the dataset from Kaggle: [Predictive Maintenance of Machines](https://www.kaggle.com/datasets/nair26/predictive-maintenance-of-machines/data). This dataset contains sensor data from industrial machines, providing valuable insights for predictive maintenance.

## Project Structure
The project consists of the following key files:
- **`app.py`**: Main Streamlit application script.
- **`/src/pm_rohithnair3.py`**: Model training script.
- **`/data/raw/CIA-1.csv`**: Original data repository.
- **`/data/processed/`**: processed data repository.
- **`RF_predictive_maintenance.pkl`**: Trained Random Forest model.
- **`scaler.pkl`**: Scaler used to normalize the dataset.
- **`medians.pkl`**: File containing the median values for dataset features.

## Installation and Execution

### 1. Clone the Repository
```bash
$ git clone https://github.com/your-repo/predictive-maintenance.git
$ cd predictive-maintenance
```

### 2. Install Dependencies
Make sure you have **Python 3.8 or higher**, then run:
```bash
$ pip install -r requirements.txt
```

### 3. Run the Application
```bash
$ streamlit run app.py
```
You can see an application using this algorithm at the following link: [Predictive Maintenance](https://predictivemaintenancerequena.streamlit.app/). 

Simply enter the three features and click the "Predict" button to determine whether a failure is predicted or not.

## How to Use the Application
1. **Adjust the sliders** to modify the following input features:
   - **Operational Hours** (Range: 5 - 170)
   - **Process Temperature** (Range: 307 - 310 °K)
   - **Air Temperature** (Range: 296 - 300 °K)

2. **Click the "Predict" button** to generate a prediction based on the trained model.
3. **Interpret the results:**
   - **Green message**: No failure predicted.
   - **Red message**: Failure predicted.

## Model Workflow
- The **Random Forest model** has been trained using preprocessed data, where **outliers were removed** and **SMOTE was applied** to balance classes.
- A **StandardScaler** is used to normalize the input features before making predictions.
- Median values from **medians.pkl** are used for features not adjustable by the user.

## Confusion Matrix Generation
The `/src/pm_rohithnair3.py` script automatically generates a **confusion matrix** to evaluate the model's performance:
```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
```
This file is saved as **confusion_matrix.png** and helps analyze the model's predictive accuracy.

## Contact
For questions or suggestions, please open an issue or contact the author at [francisco_requena@hotmail.com](mailto:francisco_requena@hotmail.com).

