
# Define functions for creating the dashboard

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


def explore_data(df):
    # Create the dashboard for exploring the dataset
    filter_widget = pn.widgets.Select(options=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])
    plot_widget = pn.widgets.Select(options=['scatter', 'histogram', 'density', 'box'])
    
    def update_plot(df, filter_by, plot_type):
        if plot_type == 'scatter':
            plot = df.hvplot.scatter(x='math score', y='reading score', c=filter_by, cmap='Category10', title='Math and Reading Scores')
        elif plot_type == 'histogram':
            plot = df.hvplot.hist(filter_by, bins=10, title='Histogram of ' + filter_by)
        elif plot_type == 'density':
            plot = df.hvplot.kde(filter_by, title='Density Plot of ' + filter_by)
        elif plot_type == 'box':
            plot = df.hvplot.box(y=filter_by, title='Box Plot of ' + filter_by)
        return plot
    
    
    @pn.depends(filter_widget.param.value, plot_widget.param.value)
    def plot(filter_by, plot_type):
       return update_plot(df, filter_by, plot_type)
    explore_dashboard = pn.Column(pn.pane.Markdown('## Explore the Dataset'), filter_widget, plot_widget, plot)
    return explore_dashboard