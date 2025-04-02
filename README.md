# Insurance Cost Analysis

[![Insurance Cost Analysis Banner](https://via.placeholder.com/800x250?text=Insurance+Cost+Analysis)](https://via.placeholder.com/800x250?text=Insurance+Cost+Analysis)

This project analyzes insurance cost data to understand the factors that influence insurance charges. It explores relationships between various features like age, BMI, smoking habits, and region, and builds a linear regression model to predict insurance costs.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Data Exploration and Visualization](#data-exploration-and-visualization)
    * [Key Visualizations](#key-visualizations)
3.  [Data Preprocessing](#data-preprocessing)
    * [Key Preprocessing Steps](#key-preprocessing-steps)
4.  [Model Building and Evaluation](#model-building-and-evaluation)
    * [Key Steps](#key-steps)
5.  [Real-Life Application](#real-life-application)
6.  [Conclusion](#conclusion)
7.  [Complete dataset, code and Output for this project](#Complete-dataset,-code-and-Output-for-this-project)

## Project Overview

This project aims to provide insights into how different factors affect insurance costs. By analyzing the data, we can identify key drivers of insurance charges and build a predictive model.

## Data Exploration and Visualization

We begin by exploring the dataset to understand the distribution of features and their relationships with insurance charges.

### Key Visualizations:

* **Scatter Plots:** To visualize the relationship between numerical features and charges.
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.scatterplot(x='bmi', y='charges', data=df)
    plt.title('Relationship between BMI and Insurance Charges')
    plt.show()
    ```
    * This code generates a scatter plot showing how Body Mass Index (BMI) relates to insurance charges.
* **Distribution Plots:** To understand the distribution of numerical features.
    ```python
    sns.histplot(df['age'], kde=True)
    plt.title('Distribution of Age')
    plt.show()
    ```
    * This code generates a histogram showing the age distribution of the dataset.
* **Box Plots:** To understand the relationship between categorical features and charges.
    ```python
    sns.boxplot(x='smoker', y='charges', data=df)
    plt.title('Insurance Charges by Smoking Status')
    plt.show()
    ```
    * This code generates a box plot showing how smoking status affects insurance charges.

## Data Preprocessing

Before building the model, we preprocess the data to make it suitable for machine learning.

### Key Preprocessing Steps:

* **One-Hot Encoding:** Convert categorical variables into numerical format.
    ```python
    df = pd.get_dummies(df, columns=['smoker', 'region'], drop_first=True)
    ```
    * This code converts the 'smoker' and 'region' columns into numerical dummy variables.
* **Scaling Numerical Features:** Standardize numerical features to have zero mean and unit variance.
    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    numerical_cols = ['age', 'bmi', 'children']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    ```
    * This code scales the 'age', 'bmi', and 'children' columns to ensure they have a similar scale.

## Model Building and Evaluation

We build a linear regression model to predict insurance charges and evaluate its performance.

### Key Steps:

* **Splitting Data:** Divide the dataset into training and testing sets.
    ```python
    from sklearn.model_selection import train_test_split

    X = df.drop('charges', axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
    * This code splits the data into training and testing sets, with 20% reserved for testing.
* **Training the Model:** Train a linear regression model on the training data.
    ```python
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)
    ```
    * This code trains a linear regression model using the training data.
* **Evaluating the Model:** Evaluate the model's performance using $R^2$ and RMSE.
    ```python
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R^2: {r2}")
    print(f"RMSE: {rmse}")
    ```
    * This code calculates and prints the $R^2$ and RMSE of the model's predictions.

## Real-Life Application

This project has practical applications in the insurance industry. By understanding the factors that influence insurance costs, insurance companies can:

* **Pricing Strategies:** Develop more accurate and fair pricing models.
* **Risk Assessment:** Identify high-risk individuals and adjust premiums accordingly.
* **Customer Insights:** Gain insights into customer behavior and preferences.
* **Policy Optimization:** Tailor insurance policies to better meet customer needs.

For example, an insurance company can use the model to predict the insurance costs for new customers based on their characteristics. This helps them to set competitive and profitable premiums.

## Conclusion

This project successfully demonstrates how data analysis and machine learning can be used to understand and predict insurance costs. By exploring the data, preprocessing it, and building a linear regression model, we gained valuable insights into the factors that influence insurance charges.

Feel free to explore the code and data in this repository. If you have any questions or suggestions, please don't hesitate to reach out.

## Jupyter Notebook

The complete code and output for this project can be found in the [IBM project.ipynb](IBM%20project.ipynb) file. GitHub renders Jupyter Notebooks, allowing you to view the code and its results directly.



## Complete dataset, code and Output for this project

The complete code and output for this project can be found in the ( https://github.com/kenny755/Insurance-Cost-Analysis/blob/master/IBM%20project.ipynb ). The code and the dataset used in this analysis can be found in the link attached.
