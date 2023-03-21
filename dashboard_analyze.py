import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import panel as pn
import hvplot.pandas
import holoviews as hv
import bokeh
import hvplot.pandas
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

hv.extension('bokeh')
pn.extension()

data = pd.read_csv("StudentsPerformance.csv")
# Check the data types of each column
print(data.dtypes)

# Check for duplicates
duplicate_rows = data[data.duplicated()]
print("Number of duplicate rows:", duplicate_rows.shape[0])
# Remove duplicates
data = data.drop_duplicates()
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)
# Fill missing values with median
data = data.fillna(data.median())

# Convert categorical variables to numerical variables
data['gender'] = data['gender'].map({'female': 0, 'male': 1})
data['race/ethnicity'] = data['race/ethnicity'].map({'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4})
data['parental level of education'] = data['parental level of education'].map({'some high school': 0, 'high school': 1, 'some college': 2, 'associate\'s degree': 3, 'bachelor\'s degree': 4, 'master\'s degree': 5})
data['lunch'] = data['lunch'].map({'standard': 0, 'free/reduced': 1})
data['test preparation course'] = data['test preparation course'].map({'none': 0, 'completed': 1})

#save the cleaned data
data.to_csv('updated.csv', index=False)
#red the new data 
df = pd.read_csv("updated.csv")

def analyze_data(df):
    # Create the dashboard for analyzing the data
    results = pd.DataFrame(columns=['Model', 'MSE', 'R2 Score'])

    # Split the dataset into training and testing sets using train_test_split from scikit-learn.
    X = df[['math score', 'reading score', 'writing score']]
    y = df['gender']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Add results to the dataframe
    results = results.append({'Model': 'Random Forest Regression', 'MSE': mse, 'R2 Score': r2}, ignore_index=True)

    # Create a plot to display the results
    plot = results.hvplot.bar(x='Model', y=['MSE', 'R2 Score'], title='Model Performance', ylim=(0, 1.1))

    # Create a dashboard with the plot
    dashboard_analyze = pn.Column(
        '# Model Performance',
        'The following plot shows the performance of the trained model on the test data.',
        plot
    )

    return dashboard_analyze

dashboard_analyze = analyze_data(df)