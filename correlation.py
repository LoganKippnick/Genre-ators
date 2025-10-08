import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Importing the dataset
data = pd.read_csv('ClassicHit.csv')
#print(data.head())

## Checking for missing values
##print(data.isnull().sum())

## Dropping rows with missing values
data = data.dropna()

## Dropping irrelevant columns
data = data.drop(columns=['Track', 'Artist'])

## Calculate the scatterplot matrix for each genre
genres = data['Genre'].unique()
for genre in genres:
    genre_data = data[data['Genre'] == genre]

    ## Creating the scatterplot matrix
    plot = sns.pairplot(genre_data, diag_kind='None', corner=True, height=2.5, kind='reg', plot_kws={'line_kws':{'color':'red'}})
    
    ## Finding Pearson correlation coefficients
    corr = genre_data.corr(method='pearson', numeric_only=True)
    #print(f'Pearson Correlation Coefficients for {genre} Genre:\n{corr}\n')
    
    plt.suptitle(f'Scatterplot Matrix for {genre} Genre', y=1.02)
    ##plt.show()
    plt.savefig(f'./scatterplots/scatterplot_matrix_{genre}.png')

    ## Create and save a table with the correlation coefficients
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title(f'Correlation Coefficients for {genre} Genre')
    plt.savefig(f'./correlationmatrices/correlation_table_{genre}.png')
    plt.close('all')


    print(f'Plots and correlation table for {genre} genre saved.')

    # Isolating lower triangle of the correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Finding pairs with high correlation
    high_corr = corr.where(~mask).stack().reset_index()
    high_corr = high_corr[high_corr[0].abs() >= 0.5]
    high_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']

    # Saving high correlation pairs to a CSV file
    high_corr.to_csv(f'./highcorrelations/high_correlation_{genre}.csv', index=False)

    # Finding pairs with medium correlation
    medium_corr = corr.where(~mask).stack().reset_index()
    medium_corr = medium_corr[(medium_corr[0].abs() < 0.5) & (medium_corr[0].abs() >= 0.3)]
    medium_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']

    # Saving medium correlation pairs to a CSV file
    medium_corr.to_csv(f'./mediumcorrelations/medium_correlation_{genre}.csv', index=False)
    print(f'High and medium correlation pairs for {genre} genre saved.\n')
    
    ##break  # Remove this break to plot for all genres