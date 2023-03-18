import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import panel as pn
import bokeh
from sklearn.model_selection import cross_val_score
import hvplot.pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import hvplot.pandas
import holoviews as hv

hv.extension('bokeh')
pn.extension()

def analyze_data(df):
    # Create the dashboard for analyzing the data
    results = pd.DataFrame(columns=['Model', 'MSE', 'R2 Score'])

    # Split the dataset into training and testing sets using train_test_split from scikit-learn.
    X = df[['math score', 'reading score', 'writing score']]
    y = df['total score']
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
    return plot

    # Create a dashboard with the plot
    dashboard_analyze = pn.Column(
        '# Model Performance',
        'The following plot shows the performance of the trained models on the test data.',
        plot
    )

    return dashboard_analyze
