
# Salary Prediction Using Simple Linear Regression

## 📌 Project Overview

This project predicts an estimated salary based on years of work experience using **Simple Linear Regression**, a supervised machine learning algorithm.

The model learns the relationship between years of experience and salary from historical data and uses that relationship to predict salaries for new inputs.

## 🎯 Objectives

- Understand and implement Simple Linear Regression.
- Load and process salary data using Pandas and NumPy.
- Train and test a machine learning model.
- Evaluate model performance using regression metrics.
- Visualize the regression line and test observations.
- Save the trained model using Pickle.
- Generate salary predictions for different years of experience.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical computations
- **Matplotlib** — Data visualization
- **Scikit-learn** — Model training and evaluation
- **Pickle** — Saving the trained model
- **VS Code** — Development environment
- **Git and GitHub** — Version control and project hosting

## 📂 Project Structure

```text
Salary-Prediction/
│
├── .venv/                         # Local virtual environment
├── Salary_Data.csv                # Dataset
├── slr.py                         # Main Python program
├── requirements.txt               # Project dependencies
├── .gitignore                     # Files excluded from Git
├── linear_regression_model.pkl    # Saved trained model
└── salary_regression_plot.png     # Regression visualization
```

The `.venv/` folder should remain local and should not be uploaded to GitHub. The model and plot are generated when the program runs.

## 📊 Dataset

The program expects a CSV file named `Salary_Data.csv` in the project folder.

Expected columns:

| YearsExperience | Salary |
| --------------: | -----: |
|             1.1 |  39343 |
|             1.3 |  46205 |
|             2.0 |  43525 |
|             2.2 |  39891 |
|             2.9 |  56642 |

*The values above are illustrative examples. Use the actual dataset for training and evaluation.*

- **Input feature (X):** Years of experience
- **Target variable (y):** Salary

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet, install the libraries directly:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 5. Run the project

```bash
python slr.py
```

## 🧠 How the Project Works

1. **Load the dataset:** Read the CSV file using Pandas.
2. **Prepare the data:** Select the experience and salary columns and remove missing rows.
3. **Split the data:** Use 80% of the data for training and 20% for testing.
4. **Train the model:** Fit a Linear Regression model to the training data.
5. **Make predictions:** Predict salaries for the test dataset.
6. **Evaluate performance:** Calculate R², MAE, MSE, and RMSE.
7. **Visualize results:** Plot test observations and the fitted regression line.
8. **Save the model:** Store the trained model in a `.pkl` file.
9. **Predict salaries:** Generate estimates for selected years of experience.

## 📐 Linear Regression Equation

The model uses the equation:

**Salary = m × YearsExperience + c**

Where:

- `m` = Slope (learned coefficient)
- `c` = Intercept
- `YearsExperience` = Input feature
- `Salary` = Predicted salary

The model learns the slope and intercept from the training data.

## 📈 Model Evaluation Metrics

- **R² Score:** Measures how well the model explains variation in the target.
- **MAE (Mean Absolute Error):** Average absolute prediction error.
- **MSE (Mean Squared Error):** Average squared prediction error.
- **RMSE (Root Mean Squared Error):** Square root of MSE, expressed in the target's units.

The program prints these metrics for both training and testing data. Actual performance depends on the dataset and the train/test split.

## 📁 Output Files

After successful execution, the program generates:

- `linear_regression_model.pkl` — Saved trained model.
- `salary_regression_plot.png` — Regression visualization.

The terminal also displays actual versus predicted salaries, evaluation metrics, and example predictions.

## ⚠️ Limitations

- Predictions depend on the quality and size of the training dataset.
- Linear Regression assumes a linear relationship between the input and target.
- Predictions outside the training data range may be unreliable.
- Salary predictions are estimates, not guaranteed salaries.
- The model has not been validated for real-world salary decisions.

## 🚀 Future Improvements

- Build a web interface using Streamlit.
- Add interactive salary prediction.
- Improve data validation and error handling.
- Compare Linear Regression with other regression algorithms.
- Deploy the application online.

## 👨‍💻 Author

**Yashwanth Balija**

GitHub: [yashwanth86868](https://github.com/yashwanth86868)

## 📄 License

This project is intended for educational and learning purposes.
