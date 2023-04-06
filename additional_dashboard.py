import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import panel as pn
import bokeh
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import hvplot.pandas
import holoviews as hv

hv.extension('bokeh')
pn.extension()

#from viz import home_page


df = pd.read_csv("updated.csv")

def additional_analysis():
    # Create the dashboard for additional analysis
    plot = df.groupby('parental level of education')['math score', 'reading score'].mean().hvplot.bar(title='Mean Scores by Parental Education Level')
    additional_dashboard = pn.Column(pn.pane.Markdown('# Additional Analysis'), plot)

    return additional_dashboard

def update_plot():
    new_plot = df.groupby('parental level of education')['writing score'].mean().hvplot.bar(title='Mean Writing Scores by Parental Education Level')
    plot_panel.object = new_plot

plot_panel = pn.panel(additional_analysis())