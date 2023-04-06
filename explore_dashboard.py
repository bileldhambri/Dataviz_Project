# Define functions for creating the dashboard

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
from sklearn.manifold import TSNE

hv.extension('bokeh')
pn.extension()


def explore_data(df):

    # Create the dashboard for exploring the dataset
    filter_widget = pn.widgets.Select(options=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])
    plot_widget = pn.widgets.Select(options=['scatter', 'histogram', 'density', 'box'])
    
    # Create a table widget to display the first 5 rows of the dataset
    table_widget = pn.widgets.Tabulator(df.head(5), height=150)
    
    # Create a matrix plot to show the correlation between the columns
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, cmap="YlGnBu", ax=ax)
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticklabels(corr.columns, rotation=0)
    corr_plot = pn.pane.Matplotlib(fig, tight=True)
    
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
    
    # Box plots of academic performance by gender
    gender_boxplot = df.hvplot.box(y=['math score', 'reading score', 'writing score'], 
              by='gender', title='Academic Performance by Gender',
              xlabel='Gender', ylabel='Score',
              color='purple', fontsize={'title': 15, 'labels': 14, 'xticks': 12, 'yticks': 12})

    # Box plots of academic performance by race/ethnicity
    race_boxplot = df.hvplot.box(y=['math score', 'reading score', 'writing score'], 
              by='race/ethnicity', title='Academic Performance by Race/Ethnicity',
              xlabel='Race/Ethnicity', ylabel='Score',
              color='green', fontsize={'title': 15, 'labels': 14, 'xticks': 12, 'yticks': 12})
    
    # Group by parental education level and calculate mean scores
    parental_edu_scores = df.groupby('parental level of education')[['math score', 'reading score', 'writing score']].mean()

    # Plot boxplots using hvplot
    parental_edu_boxplot = parental_edu_scores.hvplot.box(y=['math score', 'reading score', 'writing score'], 
    by='parental level of education', title='Academic Performance by Parental Education Level',
    xlabel='Parental Education Level', ylabel='Score',
    color='yellow', fontsize={'title': 15, 'labels': 14, 'xticks': 12, 'yticks': 12})

    #t The impact of test preparation on academic performance
    test_prep_boxplot = pn.panel(df.hvplot.box(y=['math score', 'reading score', 'writing score'], 
              by='test preparation course', title='Academic Performance by Test Preparation Completion',
              xlabel='Test Preparation Completion', ylabel='Score',
              color='orange', fontsize={'title': 15, 'labels': 14, 'xticks': 12, 'yticks': 12}))
    
    # Create box plot for academic performance by lunch type
    lunch_boxplot = df.hvplot.box(y=['math score', 'reading score', 'writing score'], 
              by='lunch', title='Academic Performance by Lunch Type',
              xlabel='Lunch Type', ylabel='Score',
              color='blue', fontsize={'title': 15, 'labels': 14, 'xticks': 12, 'yticks': 12})
    
    
    # Compute the pairwise correlation matrix
    corr_matrix = df[['math score', 'reading score', 'writing score']].corr()
    # Reduce the dimensions using t-SNE
    tsne = TSNE(n_components=2, perplexity=1, random_state=42)
    tsne_scores = tsne.fit_transform(corr_matrix)

    # Create a scatter plot with color representing correlation strength
    fig, ax = plt.subplots()
    sc = ax.scatter(df['math score'], df['reading score'], c=df['writing score'], cmap='coolwarm')
    cbar = fig.colorbar(sc)
    cbar.ax.set_ylabel('Writing Score', rotation=270, labelpad=15)
    ax.set_xlabel('Math Score')
    ax.set_ylabel('Reading Score')
    ax.set_title('Scatter Plot of Test Scores')


    # Wrap the scatter plot in a Panel object
    corr_tsne_plot = pn.pane.Matplotlib(fig, tight=True)
    tsne_row = pn.Row(corr_tsne_plot)
    
    explore_dashboard = pn.Column(
        pn.pane.Markdown("""
        ## Explore the Dataset
        
        This dataset contains information about students' performance in math, reading, and writing exams. There are five categories of data included: gender, race/ethnicity, parental level of education, lunch, and test preparation course. 
        
        The purpose of this dashboard is to help you explore the dataset and gain insights into the relationships between different variables. You can use the dropdown menus to select which variable to filter by and which type of plot to display. 
        
        ## Why is this dataset important?
        
        Understanding the factors that influence student performance is crucial for improving educational outcomes. This dataset provides a valuable resource for identifying which factors are most strongly associated with academic success, and how different groups of students may experience education differently.
        """),
        pn.Row(
            pn.Column(
                table_widget,
                width=400
            )),
        filter_widget,
        plot_widget,
        plot,
        gender_boxplot,
        race_boxplot,
        parental_edu_boxplot,
        test_prep_boxplot,
        lunch_boxplot,
        corr_tsne_plot
        
        )
    #explore_dashboard.append(tsne_row)

    
    return explore_dashboard