# Student Depression Analysis Dashboard

This project focuses on analyzing mental health and depression among students using machine learning and deep learning models. It provides a fully interactive and visual dashboard that allows users to explore insights, visualize results, and make predictions based on a given dataset.

## 📁 Project Structure

- `data/`: Raw and cleaned datasets
- `models/`: Trained ML/DL models
- `notebooks/`: Jupyter notebooks for EDA and model building
- `dashboard/`: Interactive dashboard UI and logic (Streamlit or similar)
- `README.md`: Overview and documentation of the project

## 🎯 Objective

The primary goal of this project is to:
- Understand student mental health trends
- Predict likelihood of depression based on key indicators
- Visualize relationships among features and their impact on mental health
- Provide decision support to institutions and psychologists

## 🧠 Machine Learning Models Used

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost
- Neural Networks (TensorFlow/Keras)
- AutoML (TPOT or LightAutoML)

Each model was trained and evaluated using accuracy, precision, recall, F1-score, and ROC-AUC metrics. The best-performing models were selected for integration into the dashboard.

## 📊 Dashboard Features

- **Home Page**: Project summary, description, and navigation.
- **Upload Dataset**: Allows users to upload their CSV file for analysis.
- **Visualize Data**: Correlation matrix, bar plots, pie charts, heatmaps, and more.
- **Model Training**: Train selected models on the uploaded dataset.
- **Model Comparison**: Performance comparison using graphs and tables.
- **Prediction Page**: User input fields for real-time depression prediction.
- **Result Analysis**: ROC Curve, Confusion Matrix, Feature Importance.
- **Download Reports**: Export results and analysis as PDF or CSV.

## 📚 Dataset Description

- Source: [Kaggle Mental Health Dataset](https://www.kaggle.com/)
- Fields: Gender, Age, Stress level, Anxiety level, Sleep quality, Academic pressure, Suicidal thoughts, Family support, etc.
- Rows: ~1000
- Target Variable: `Depression` (Binary: Yes/No)

## 🧹 Preprocessing Steps

- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Feature selection and engineering
- Train-test splitting

## ⚙️ Tech Stack

- **Frontend**: Streamlit / PyQt5
- **Backend**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib, Seaborn, Plotly, LightAutoML
- **Visualization**: seaborn, plotly, matplotlib, streamlit

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/student-depression-analysis.git
cd student-depression-analysis

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
