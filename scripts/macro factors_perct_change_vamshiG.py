# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 00:46:34 2023

@author: vamsh
"""

import pandas as pd

# Path to CSV file
csv_file = 'C:\\Users\\vamsh\\Documents\\Uni\\UChicago\\Summer 2023\\MSCA 34003_IP01 - Capstone II\\Project\\Data\\Macro Factors\\'

#### Macro-economic Factors

merged_table_final_filtered = pd.read_csv(csv_file + 'merged_table_final_filtered.csv')

# new dataframe to store the percentage changes
percentage_changes_df = pd.DataFrame()

# Include the DATE column in the new dataframe
percentage_changes_df['DATE'] = merged_table_final_filtered['DATE']

# Calculate percentage changes for each column except DATE
columns_to_calculate = ['Average_MSRP_spec', 'CCI', 'GDP', 'Metal', 'Iron_Ore', 'Inflation']
for column in columns_to_calculate:
    percentage_changes_df[column] = merged_table_final_filtered[column].pct_change()

# Omit the first row since it's NaN
percentage_changes_df = percentage_changes_df.iloc[1:]

# Saving the data in a new dataframe
merged_table_final_filtered_perct_change_macro1 = percentage_changes_df.copy()

# Save the DataFrame to a CSV file
# merged_table_final_filtered_perct_change_macro1.to_csv(csv_file + 'merged_table_final_filtered_perct_change_macro1.csv', index=False)

# Load DataFrame
# merged_table_final_filtered_perct_change_macro1 = pd.read_csv(csv_file + 'merged_table_final_filtered_perct_change_macro1_6month_lag.csv')


import seaborn as sns
import matplotlib.pyplot as plt

# Selecting the columns for scatterplots, correlation heatmap, and correlation calculations
columns_for_correlation = ['Average_MSRP_spec', 'CCI', 'GDP', 'Metal', 'Iron_Ore', 'Inflation']

# Correlation Heatmap
correlation_data = merged_table_final_filtered_perct_change_macro1[columns_for_correlation]
correlation_matrix = correlation_data.corr()
# Plotting Pearson's Correlation Heatmap
# The scores are between -1 and 1 for perfectly negatively correlated variables and perfectly positively 
# correlated respectively
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Pearson's Correlation Heatmap of % change in Macro Factors and average MSRP")
plt.show()

# Pearson's Correlation table
pearson_correlation = correlation_data.corr(method='pearson')
pearson_table = pd.DataFrame(pearson_correlation, columns=columns_for_correlation, index=columns_for_correlation)
print(pearson_table)

# Plotting Spearman's Correlation Heatmap
# As with the Pearson correlation coefficient, the scores are between -1 and 1
spearman_correlation = correlation_data.corr(method='spearman')
sns.heatmap(spearman_correlation, annot=True, cmap='coolwarm')
plt.title("Spearman's Correlation Heatmap of % change in Macro Factors and average MSRP")
plt.show()

# Spearman's Correlation table
spearman_correlation = correlation_data.corr(method='spearman')
spearman_table = pd.DataFrame(spearman_correlation, columns=columns_for_correlation, index=columns_for_correlation)
print(spearman_table)


#### PPI Commodity data

merged_table_final_filtered = pd.read_csv(csv_file + 'merged_table_final_filtered_Products.csv')

# new dataframe to store the percentage changes
percentage_changes_df = pd.DataFrame()

# Include the DATE column in the new dataframe
percentage_changes_df['DATE'] = merged_table_final_filtered['DATE']

# Calculate percentage changes for each column except DATE
columns_to_calculate = ['Average_MSRP_spec', 'Semi_Price_Index', 'WPU03THRU15_Price_Index', 'WPU10_Price_Index', 'WPU1017_Price_Index']
for column in columns_to_calculate:
    percentage_changes_df[column] = merged_table_final_filtered[column].pct_change()

# Omit the first row since it's NaN
percentage_changes_df = percentage_changes_df.iloc[1:]

# Saving the data in a new dataframe
merged_table_final_filtered_perct_change_macro2 = percentage_changes_df.copy()

# Save the DataFrame to a CSV file
# merged_table_final_filtered_perct_change_macro2.to_csv(csv_file + 'merged_table_final_filtered_perct_change_macro2.csv', index=False)

# Load DataFrame
# merged_table_final_filtered_perct_change_macro2 = pd.read_csv(csv_file + 'merged_table_final_filtered_perct_change_macro2_3month_lag.csv')


import seaborn as sns
import matplotlib.pyplot as plt

# Selecting the columns for scatterplots, correlation heatmap, and correlation calculations
columns_for_correlation = ['Average_MSRP_spec', 'Semi_Price_Index', 'WPU03THRU15_Price_Index', 'WPU10_Price_Index', 'WPU1017_Price_Index']

# Correlation Heatmap
correlation_data = merged_table_final_filtered_perct_change_macro2[columns_for_correlation]
correlation_matrix = correlation_data.corr()
# Plotting Pearson's Correlation Heatmap
# The scores are between -1 and 1 for perfectly negatively correlated variables and perfectly positively 
# correlated respectively
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Pearson's Correlation Heatmap of % change in Product Macro Factors and average MSRP")
plt.show()

# Pearson's Correlation table
pearson_correlation = correlation_data.corr(method='pearson')
pearson_table = pd.DataFrame(pearson_correlation, columns=columns_for_correlation, index=columns_for_correlation)
print(pearson_table)

# Plotting Spearman's Correlation Heatmap
# As with the Pearson correlation coefficient, the scores are between -1 and 1
spearman_correlation = correlation_data.corr(method='spearman')
sns.heatmap(spearman_correlation, annot=True, cmap='coolwarm')
plt.title("Spearman's Correlation Heatmap of % change in Product Macro Factors and average MSRP")
plt.show()

# Spearman's Correlation table
spearman_correlation = correlation_data.corr(method='spearman')
spearman_table = pd.DataFrame(spearman_correlation, columns=columns_for_correlation, index=columns_for_correlation)
print(spearman_table)

