import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

df = pd.read_csv("updated.csv")

# import the three functions that generate the dashboards
from dashboard_analyze import analyze_data
from additional_dashboard import additional_analysis
from explore_dashboard import explore_data

# Define the home page with buttons that lead to each dashboard
explore_button = pn.widgets.Button(name="Exploring data", button_type="primary")
analyze_button = pn.widgets.Button(name="Model Performance", button_type="primary")
additional_button = pn.widgets.Button(name="Additional Analysis", button_type="primary")

home_page = pn.Column(
    pn.pane.Markdown("#### Home Page"),
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

def show_dashboard_explore(event):
    home_page[-1] = explore_data(df)

def show_dashboard_analyze(event):
    plot = analyze_data(df)
    dashboard_analyze = pn.Column(
        plot
    )
    home_page[-1] = dashboard_analyze

def show_additional_dashboard(event):
    home_page[-1] = additional_analysis()

# Attach the actions to the buttons
home_page[1][0][0].param.watch(show_dashboard_explore, "clicks")
home_page[1][0][0].callback_args = {'event': None}
home_page[1][1][0].param.watch(show_dashboard_analyze, "clicks")
home_page[1][1][0].callback_args = {'event': None}
home_page[1][2][0].param.watch(show_additional_dashboard, "clicks")
home_page[1][2][0].callback_args = {'event': None}


# Serve the home page
home_page.servable()