import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
altura_en_m = df['height'] / 100
df['BMI'] = (df['weight']) / (altura_en_m ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

mask = df[['cholesterol', 'gluc']] == 1
df[mask] = 0

mask = df[['cholesterol', 'gluc']] > 1
df[mask] = 1

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars = ['cardio'], value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = None

    # Draw the catplot with 'sns.catplot()'



    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    mask_diastolic = df['ap_lo'] <= df['ap_hi']

    mask_height_1 = df['height'] >= df['height'].quantile(0.025)
    mask_height_2 = df['height'] <= df['height'].quantile(0.975)

    mask_weight_1 = df['weight'] >= df['weight'].quantile(0.025)
    mask_weight_2 = df['weight'] <= df['weight'].quantile(0.975)

    mask_total = (mask_diastolic) & (mask_height_1) & (mask_height_2) & (mask_weight_1) & (mask_weight_2)

    df_heat = df[mask_total]

    # Calculate the correlation matrix
    corr = None

    # Generate a mask for the upper triangle
    mask = None



    # Set up the matplotlib figure
    fig, ax = None

    # Draw the heatmap with 'sns.heatmap()'



    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
