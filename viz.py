##!pip install pandas numpy matplotlib seaborn scikit-learn panel bokeh hvplot

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


df = pd.read_csv("StudentsPerformance.csv")
# Check the data types of each column
print(df.dtypes)

# Check for duplicates
duplicate_rows = df[df.duplicated()]
print("Number of duplicate rows:", duplicate_rows.shape[0])

# Remove duplicates
df = df.drop_duplicates()

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Fill missing values with median
df = df.fillna(df.median())

# Convert categorical variables to numerical variables
df['gender'] = df['gender'].map({'female': 0, 'male': 1})
df['race/ethnicity'] = df['race/ethnicity'].map({'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4})
df['parental level of education'] = df['parental level of education'].map({'some high school': 0, 'high school': 1, 'some college': 2, 'associate\'s degree': 3, 'bachelor\'s degree': 4, 'master\'s degree': 5})
df['lunch'] = df['lunch'].map({'standard': 0, 'free/reduced': 1})
df['test preparation course'] = df['test preparation course'].map({'none': 0, 'completed': 1})


# import the three functions that generate the dashboards
from dashboard_analyze import analyze_data
from additional_dashboard import additional_analysis
from explore_dashboard import explore_data

# Define the home page with buttons that lead to each dashboard
explore_button = pn.widgets.Button(name="Exploring data", button_type="primary")
analyze_button = pn.widgets.Button(name="Model Performance", button_type="primary")
additional_button = pn.widgets.Button(name="Additional Analysis", button_type="primary")

home_page = pn.Column(
    pn.pane.Markdown("## Home Page"),
    pn.Row(
        pn.Column(
            explore_button,
            pn.pane.Markdown(),
            width=200
        ),
        pn.Column(
            analyze_button,
            pn.pane.Markdown(),
            width=200
        ),
        pn.Column(
            additional_button,
            pn.pane.Markdown(),
            width=200
        ),
        align="center"
    )
)


# Define the actions to perform when each button is clicked
"""def show_dashboard_explore(event):
    # Create the explore dashboard
    explore_dashboard = explore_data(df)
    explore_dashboard.show()
def show_dashboard_analyze(event):
    # Create the analyze dashboard
    dashboard_analyze = analyze_data(df)
    dashboard_analyze.show()
def show_additional_dashboard(event):
    # Create the additional analysis dashboard
    additional_dashboard = additional_analysis(df)
    additional_dashboard.show()"""

def show_dashboard_explore(event):
    home_page[-1] = explore_data(df)

def show_dashboard_analyze(event):
    home_page[-1] = analyze_data(df)

def show_additional_dashboard(event):
    home_page[-1] = additional_analysis()


# Attach the actions to the buttons
#explore_button.on_click(show_dashboard_explore)
#analyze_button.on_click(show_dashboard_analyze)
#additional_button.on_click(show_additional_dashboard)

# Attach the actions to the buttons
home_page[1][0][0].param.watch(show_dashboard_explore, "clicks")
home_page[1][0][0].callback_args = {'event': None}
home_page[1][1][0].param.watch(show_dashboard_analyze, "clicks")
home_page[1][1][0].callback_args = {'event': None}
home_page[1][2][0].param.watch(show_additional_dashboard, "clicks")
home_page[1][2][0].callback_args = {'event': None}


# Serve the home page
home_page.servable()






