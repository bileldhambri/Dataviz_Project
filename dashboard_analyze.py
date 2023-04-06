import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import panel as pn
import hvplot.pandas
import holoviews as hv
from sklearn.cluster import KMeans
import bokeh
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

pn.extension('tabulator')

#from viz import home_page


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

    # Train and evaluate the models
    models = {
        'RF Regression': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
        'SVR': SVR(kernel='linear'),
        'K-NN Regression': KNeighborsRegressor(n_neighbors=5)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Add results to the dataframe
        results = results.append({'Model': model_name, 'MSE': mse, 'R2 Score': r2}, ignore_index=True)
    
    # Create a plot to display the results
    plot = results.hvplot.bar(x='Model', y=['MSE', 'R2 Score'], title='Model Performance', ylim=(0, 1.1))
    
    # Create a table to display the results
    table = pn.widgets.Tabulator(results, layout='fit_data_stretch', height=300, selectable=True)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    df['Cluster Label'] = kmeans.labels_
    
    # Create a scatter plot to display the clustering results
    cluster_plot = df.hvplot.scatter(x='math score', y='reading score', by='Cluster Label', cmap='Category10', title='Cluster Analysis')
   
    # Create plots to display the results of each model
    model_plots = {}
    for model_name, model in models.items():
        model_plots[model_name] = pd.DataFrame({'Actual': y_test, 'Predicted': model.predict(X_test)}).hvplot.scatter(x='Actual', y='Predicted', title=model_name)
    
     # Create a function to print the best model based on MSE
    def best_model_mse():
        best_mse = results.loc[results['MSE'].idxmin()]
        return f"The best model based on MSE is {best_mse['Model']} with an MSE of {best_mse['MSE']:.2f}."
    
     # Define a function to find the best model based on R2 Score
    def best_model_r2():
        best_r2 = results.loc[results['R2 Score'].idxmax()]
        return f"The best model based on R2 Score is {best_r2['Model']} with an R2 Score of {best_r2['R2 Score']:.2f}."
    
    # Add the best models based on MSE and R2 Score to the dashboard
    dashboard_analyze = []
    best_mse = best_model_mse()
    best_r2 = best_model_r2()
    dashboard_analyze.append(pn.Row(pn.Column(best_mse), pn.Column(best_r2)))

    # Create a dashboard with the plot and table
    dashboard_analyze = pn.Column(
        '# Model Performance',
        'The following plot shows the performance of the trained models on the test data.',
        plot,
        'Model Performance Table',
        table,
        '# Best Model based on MSE',
        best_mse,
        '# Best Model based on R2',
        best_r2,
        '# Cluster Analysis',
        'The following plot shows the clusters of similar students based on their performance in math and reading.',
        pn.Column(cluster_plot),
        '# Model Predictions',
        'The following plots show the predicted versus actual scores for each model.',
        pn.Column(*list(model_plots.values())
        ))
    
    return dashboard_analyze

dashboard_analyze = analyze_data(df)