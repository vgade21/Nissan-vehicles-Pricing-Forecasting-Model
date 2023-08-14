# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:24:23 2023

@author: vamsh
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

csv_file = 'C:\\Users\\vamsh\\Documents\\Uni\\UChicago\\Summer 2023\\MSCA 34003_IP01 - Capstone II\\Project\\Data\\Macro Factors\\'
df_1 = pd.read_csv(csv_file + 'merged_table_final_all_macro.csv')
csv_file_2 = 'C:\\Users\\vamsh\\Documents\\Uni\\UChicago\\Summer 2023\\MSCA 34003_IP01 - Capstone II\\Project\\Data\\'
df_2 = pd.read_csv(csv_file_2 + 'df_for_modeling.csv')
csv_file_3 = 'C:\\Users\\vamsh\\Documents\\Uni\\UChicago\\Summer 2023\\MSCA 34003_IP01 - Capstone II\\Project\\script\\Mixed Effects model\\'

dataset_macro = df_1.copy()
dataset_MSRP = df_2.copy()

# Merging the MSRP data and Macro Factor data
# Convert the date columns to the desired format
dataset_macro['DATE'] = pd.to_datetime(dataset_macro['DATE']).dt.strftime('%Y-%m')
dataset_MSRP['PROFILE_Data_date_2'] = pd.to_datetime(dataset_MSRP['PROFILE_Data date']).dt.strftime('%Y-%m')
# Left join merge on the two dataframes
merged_df = pd.merge(dataset_MSRP, dataset_macro, how='left', left_on='PROFILE_Data_date_2', right_on='DATE')

# Adding the suffix "_Macro" to all the macro factor columns and moving them next to the "TARGET_Retail price including delivery charge" column
# Index of the "PROFILE_Data_date_2" column
target_index = merged_df.columns.get_loc("PROFILE_Data_date_2")
target_index_2 = merged_df.columns.get_loc("TARGET_Retail price including delivery charge")
# Get the columns to be renamed and moved
columns_to_rename = merged_df.columns[target_index+1:]
# Rename the columns with the suffix "_Macro"
new_column_names = {col: col + "_Macro" for col in columns_to_rename}
merged_df = merged_df.rename(columns=new_column_names)
# Get the column names of the merged_df dataframe
column_names = list(merged_df.columns)
# Identify the last 11 columns
last_columns = column_names[-11:]
# Reorder the columns by moving the last 11 columns next to the target column
new_column_order = column_names[:target_index_2 + 1] + last_columns + column_names[target_index_2 + 1:-11]
# Reindex the merged_df dataframe with the new column order
merged_df = merged_df.reindex(columns=new_column_order)

# Dropping unnecessary columns
# Define the columns to be dropped
columns_to_drop = ['PROFILE_Data_date_2', 'DATE_Macro', 'Average_MSRP_spec_Macro']
# Drop the columns from the merged_df dataframe
merged_df = merged_df.drop(columns=columns_to_drop)

df_for_modeling_all_macro_final = merged_df.copy()
# Save the DataFrame to a CSV file
# df_for_modeling_all_macro_final.to_csv(csv_file_2 + 'df_for_modeling_all_macro_final.csv', index=False)
# Load DataFrame
# df_for_modeling_all_macro_final = pd.read_csv(csv_file_2 + 'df_for_modeling_all_macro_final.csv')

# Missing data in the Macro columns
# Filter columns that end with "_Macro"
macro_columns = df_for_modeling_all_macro_final.columns[df_for_modeling_all_macro_final.columns.str.endswith('_Macro')]
# Calculate the count of missing data in each macro column
missing_data_count = df_for_modeling_all_macro_final[macro_columns].isnull().sum()
# Print the count of missing data in each macro column
'''
All columns except Inflation_Macro: 19 missing data (2012 data)
Inflation_Macro: 1345 missing data (2012 + 2022-05 - 2023-03 data)
'''
for column, count in missing_data_count.items():
    print(f"Column '{column}': {count} missing data")

# Removing rows with dates in 2012 and 2022-05 onwards in the "PROFILE_Data date" column
# Convert the "PROFILE_Data date" column to datetime format
df_for_modeling_all_macro_final['PROFILE_Data date'] = pd.to_datetime(df_for_modeling_all_macro_final['PROFILE_Data date'])
# Filter the rows based on the specified date conditions
df_for_modeling_all_macro_final = df_for_modeling_all_macro_final[(df_for_modeling_all_macro_final['PROFILE_Data date'].dt.year != 2012) & (df_for_modeling_all_macro_final['PROFILE_Data date'] < '2022-05-01')]
# Reset the index of the dataframe
df_for_modeling_all_macro_final = df_for_modeling_all_macro_final.reset_index(drop=True)

df_for_modeling_all_macro_final_filtered = df_for_modeling_all_macro_final.copy()
# Save the DataFrame to a CSV file
# df_for_modeling_all_macro_final_filtered.to_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered.csv', index=False)
# Load DataFrame
# df_for_modeling_all_macro_final_filtered = pd.read_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered.csv')

# Missing data in the Macro columns
# Calculate the count of missing data in each macro column
missing_data_count = df_for_modeling_all_macro_final_filtered[macro_columns].isnull().sum()
# Print the count of missing data in each macro column
'''
All Macro columns have 0 missing data now
'''
for column, count in missing_data_count.items():
    print(f"Column '{column}': {count} missing data")

# Missing data in the all columns
# Check for missing values in each row
missing_rows = df_for_modeling_all_macro_final_filtered.isnull().any(axis=1)
# Count the number of rows with at least one missing value
num_missing_rows = missing_rows.sum()
# Print the number of rows with at least one missing value
'''
1416/13012 rows with at least one column with missing data
'''
print(f"Number of rows with missing data: {num_missing_rows}")

df_for_modeling_all_macro_final_filtered_cleaned = df_for_modeling_all_macro_final_filtered.copy()

# Cleaning the dataset, encoding and dropping columns
# Replacing FALSE with 0 and TRUE with 1 in column "SPEC_DC1socket"
df_for_modeling_all_macro_final_filtered_cleaned["SPEC_DC1socket"] = df_for_modeling_all_macro_final_filtered_cleaned["SPEC_DC1socket"].replace({False: 0, True: 1})
# Replacing FALSE with 0 and TRUE with 1 in column "SPEC_AC1socket"
df_for_modeling_all_macro_final_filtered_cleaned["SPEC_AC1socket"] = df_for_modeling_all_macro_final_filtered_cleaned["SPEC_AC1socket"].replace({False: 0, True: 1})
# Removing rows with specific entries in "PROFILE_Trim level" column
entries_to_remove = ["-", "+", "!"]
df_for_modeling_all_macro_final_filtered_cleaned = df_for_modeling_all_macro_final_filtered_cleaned[~df_for_modeling_all_macro_final_filtered_cleaned["PROFILE_Trim level"].isin(entries_to_remove)]
# Dropping unnecessary columns
columns_to_drop = ["PROFILE_Name", "PROFILE_Unique Identity", "PROFILE_Data date", "PROFILE_Version state", "PROFILE_Data status", "TARGET_Retail price including delivery charge"]
df_for_modeling_all_macro_final_filtered_cleaned = df_for_modeling_all_macro_final_filtered_cleaned.drop(columns=columns_to_drop)
# Moving the "TARGET_Retail price" column to the front of the dataframe

# Moving the "TARGET_" columns to the front of the table
# Define the columns to be moved to the front
columns_to_move = ["TARGET_Retail price", "TARGET_Volume", "TARGET_Weighted_RetailPrice", "TARGET_VolumebyModel"]
# Reorder the columns by moving the specified columns to the front
new_column_order = columns_to_move + [col for col in df_for_modeling_all_macro_final_filtered_cleaned.columns if col not in columns_to_move]
# Reindex the dataframe with the new column order
df_for_modeling_all_macro_final_filtered_cleaned = df_for_modeling_all_macro_final_filtered_cleaned.reindex(columns=new_column_order)

# Moving the "Segment_Competing_Nissan_Model" and  "Segment_Name" columns next to column "PROFILE_JATO regional segment"
# Identify the index of the "PROFILE_JATO regional segment" column
jato_segment_index = df_for_modeling_all_macro_final_filtered_cleaned.columns.get_loc("PROFILE_JATO regional segment")
# Reorder the columns with "Segment_Competing_Nissan_Model" and "Segment_Name" next to "PROFILE_JATO regional segment"
new_column_order = list(df_for_modeling_all_macro_final_filtered_cleaned.columns)
new_column_order.remove("Segment_Competing_Nissan_Model")
new_column_order.remove("Segment_Name")
new_column_order.insert(jato_segment_index + 1, "Segment_Competing_Nissan_Model")
new_column_order.insert(jato_segment_index + 2, "Segment_Name")
# Reindex the dataframe with the new column order
df_for_modeling_all_macro_final_filtered_cleaned = df_for_modeling_all_macro_final_filtered_cleaned[new_column_order]

# Save the DataFrame to a CSV file
# df_for_modeling_all_macro_final_filtered_cleaned.to_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered_cleaned.csv', index=False)
# Load DataFrame
# df_for_modeling_all_macro_final_filtered_cleaned = pd.read_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered_cleaned.csv')

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import statsmodels.formula.api as smf

df_for_modeling_all_macro_final_filtered_cleaned_2 = df_for_modeling_all_macro_final_filtered_cleaned.copy()
# Dropping the "TARGET_Retail price" column since we have "TARGET_Weighted_RetailPrice"
df_for_modeling_all_macro_final_filtered_cleaned_2 = df_for_modeling_all_macro_final_filtered_cleaned_2.drop(columns=["TARGET_Retail price"])

df_for_modeling_all_macro_final_filtered_rename_columns = df_for_modeling_all_macro_final_filtered_cleaned_2.copy()
# Replace characters in column names
df_for_modeling_all_macro_final_filtered_rename_columns.columns = df_for_modeling_all_macro_final_filtered_rename_columns.columns.str.replace(' ', '_')
df_for_modeling_all_macro_final_filtered_rename_columns.columns = df_for_modeling_all_macro_final_filtered_rename_columns.columns.str.replace(',', '_')
df_for_modeling_all_macro_final_filtered_rename_columns.columns = df_for_modeling_all_macro_final_filtered_rename_columns.columns.str.replace('/', '_')
df_for_modeling_all_macro_final_filtered_rename_columns.columns = df_for_modeling_all_macro_final_filtered_rename_columns.columns.str.replace('&', '_')
df_for_modeling_all_macro_final_filtered_rename_columns.columns = df_for_modeling_all_macro_final_filtered_rename_columns.columns.str.replace('.', '_')
df_for_modeling_all_macro_final_filtered_rename_columns.columns = df_for_modeling_all_macro_final_filtered_rename_columns.columns.str.replace('-', '_')
df_for_modeling_all_macro_final_filtered_rename_columns.columns = df_for_modeling_all_macro_final_filtered_rename_columns.columns.str.replace(')', '_')
df_for_modeling_all_macro_final_filtered_rename_columns.columns = df_for_modeling_all_macro_final_filtered_rename_columns.columns.str.replace('(', '_')
df_for_modeling_all_macro_final_filtered_rename_columns.columns = df_for_modeling_all_macro_final_filtered_rename_columns.columns.str.replace('~', '_')
df_for_modeling_all_macro_final_filtered_rename_columns.columns = df_for_modeling_all_macro_final_filtered_rename_columns.columns.str.replace("'", '_')

# Count the number of rows with missing data
num_rows_with_missing_data = df_for_modeling_all_macro_final_filtered_rename_columns.isnull().any(axis=1).sum()
print("Number of rows with at least one column containing missing data:", num_rows_with_missing_data)

# Find the index of the first three rows with missing data
rows_with_missing_data = df_for_modeling_all_macro_final_filtered_rename_columns[df_for_modeling_all_macro_final_filtered_rename_columns.isnull().any(axis=1)].head(3).index
print("Index of the first three rows with at least one column containing missing data:")
print(rows_with_missing_data)

df_for_modeling_all_macro_final_filtered_cleaned_rename_delete = df_for_modeling_all_macro_final_filtered_rename_columns.copy()
# Delete rows with missing data
df_for_modeling_all_macro_final_filtered_cleaned_rename_delete = df_for_modeling_all_macro_final_filtered_cleaned_rename_delete.dropna()
# Replace characters in column names
df_for_modeling_all_macro_final_filtered_cleaned_rename_delete.columns = df_for_modeling_all_macro_final_filtered_cleaned_rename_delete.columns.str.replace("___", '_')
# Replace trailing underscores in column names
df_for_modeling_all_macro_final_filtered_cleaned_rename_delete.columns = [col.rstrip('_') for col in df_for_modeling_all_macro_final_filtered_cleaned_rename_delete.columns]
# Dropping "SPEC_Compressor_turbo" column since it's duplicated
##################
####### Make sure to manually delete one of the "SPEC_Compressor_turbo" columns
##################
# Count the number of rows with missing data
num_rows_with_missing_data = df_for_modeling_all_macro_final_filtered_cleaned_rename_delete.isnull().any(axis=1).sum()
print("Number of rows with at least one column containing missing data:", num_rows_with_missing_data)

# Save the DataFrame to a CSV file
# df_for_modeling_all_macro_final_filtered_cleaned_rename_delete.to_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered_cleaned_rename_delete.csv', index=False)
# Load DataFrame
# df_for_modeling_all_macro_final_filtered_cleaned_rename_delete = pd.read_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered_cleaned_rename_delete.csv')


from sklearn.model_selection import KFold
import numpy as np

def custom_cross_val_score(formula, data, groups, k=5):
    """Calculate the cross-validation R^2 for a mixed-effects model."""
    # Define the K-fold cross-validator
    kf = KFold(n_splits=k)

    r2_scores = []
    
    # Loop over each fold
    for train_index, test_index in kf.split(data):
        # Define the training and test sets
        data_train, data_test = data.iloc[train_index].copy(), data.iloc[test_index].copy()

        # Define the groups for the mixed-effects model
        groups = data_train['PROFILE_Make'] + "_" + data_train['CATEGORY']

        # Fit the mixed-effects model
        model = smf.mixedlm(formula, data_train, groups=groups).fit()

        # Predict the target variable for the test set
        data_test.loc[:, 'TARGET_Weighted_RetailPrice_pred'] = model.predict(data_test)

        # Calculate the Test R^2 and append to the list
        r2_scores.append(r2_score(data_test['TARGET_Weighted_RetailPrice'], data_test['TARGET_Weighted_RetailPrice_pred']))

    return np.mean(r2_scores)


# Load the data
# df = df_for_modeling_all_macro_final_filtered_cleaned_rename_delete.copy()
# 85 variables (selected using Tree based algorithms) with CATEGORY (cluster groups) and no dummy variables
df = pd.read_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered_cleaned_rename_delete_intersection_CATEGORY.csv')

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

# Define the target variable and the features
target = 'TARGET_Weighted_RetailPrice'
# all features less the target variable and grouping variables
features = df.columns.drop([target, 'PROFILE_Make', 'CATEGORY'])

# Split the dataset into a training set and a test set
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Check the sizes of the training set and the test set
df_train.shape, df_test.shape


# Defining the formula for the mixed-effects model
formula = target + ' ~ ' + ' + '.join(features)

# Define the groups for the mixed-effects model
'''
In the context of a mixed-effects model the grouping variable defines the structure of the random effects. 
Each unique value of the grouping variable corresponds to a different group. In this case, we want to 
account for the random effects at the level of each unique combination of 'PROFILE_Make' and 
'CATEGORY'. By concatenating the two columns, we effectively create a new identifier that is unique 
for each combination of 'PROFILE_Make' and 'CATEGORY'.
'''
# Note: concatenation is used here to create a unique identifier for each group defined by the combination 
# of 'PROFILE_Make' and 'CATEGORY'. This allows the mixed-effects model to estimate a separate 
# random effect for each group.
groups = df_train['PROFILE_Make'] + "_" + df_train['CATEGORY']

# Fit the mixed-effects model
# using the mixedlm function from the statsmodels library as it allows us to specify both fixed effects 
# (the usual regression coefficients) and random effects (the group-level effects)
# Note: model's convergence warning suggests that the model might not have converged to the best solution
# and the results should be interpreted with caution.
mixed_model = smf.mixedlm(formula, df_train, groups=groups).fit()

# Show the model summary
# Note: A small p-value (≤ 0.05) indicates strong evidence that the coefficient is different from zero,
# strong evidence that the corresponding variable has an effect on the TARGET_Weighted_RetailPrice.
# all features expect 13 are statistically significant based on this model
mixed_model.summary()


# Predictions on the test set using the fitted model
# Predict the target variable for the test set
df_test['TARGET_Weighted_RetailPrice_pred'] = mixed_model.predict(df_test)
df_train['TARGET_Weighted_RetailPrice_pred'] = mixed_model.predict(df_train)

# Show the first few rows of the test set
df_test[['TARGET_Weighted_RetailPrice', 'TARGET_Weighted_RetailPrice_pred']].head()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate the Mean Absolute Error (MAE)
# MAE tells us that on averag our model's predictions are approximately $6525.28 off from the actual values.
mae = mean_absolute_error(df_test['TARGET_Weighted_RetailPrice'], df_test['TARGET_Weighted_RetailPrice_pred'])

# Calculate the Mean Squared Error (MSE)
# used to calculate RMSE
mse_test = mean_squared_error(df_test['TARGET_Weighted_RetailPrice'], df_test['TARGET_Weighted_RetailPrice_pred'])
mse_train = mean_squared_error(df_train['TARGET_Weighted_RetailPrice'], df_train['TARGET_Weighted_RetailPrice_pred'])

# Calculate the Coefficient of Determination (R^2)
# R^2 of 0.7803 suggests that the model explains approximately 78.03% of the variance in the test set 
# target variable.
r2_test = r2_score(df_test['TARGET_Weighted_RetailPrice'], df_test['TARGET_Weighted_RetailPrice_pred'])
r2_train = r2_score(df_train['TARGET_Weighted_RetailPrice'], df_train['TARGET_Weighted_RetailPrice_pred'])

# Calculate the Root Mean Squared Error (RMSE)
# So on average the model's predictions are around $9486.85 off from the actual values when considering 
# both underpredictions and overpredictions.
rmse_test = np.sqrt(mse_test)
rmse_train = np.sqrt(mse_train)

# Now apply this function on your data
cv_r2 = custom_cross_val_score(formula, df_train, groups)

mae, mse_test, mse_train, r2_test, r2_train, rmse_test, rmse_train, cv_r2

'''
mae, mse_test, mse_train, r2_test, r2_train, rmse_test, rmse_train, cv_r2
Out[68]: 
(4449.9411475387615,
 42060599.942561105,
 40176094.06734045,
 0.897339249835174,
 0.8949442720729494,
 6485.41440021847,
 6338.461490562237,
 0.8905541706958869)
'''

# Load the dataset
# dataset with 85 features with CATEGORY (cluster groups) and dummy variables
df_new = pd.read_csv(csv_file_3 + 'df_for_MixedEffectModel_intersection.csv')

# Define the target variable and the features
target_new = 'TARGET_Weighted_RetailPrice'
features_new = df_new.columns.drop([target_new, 'CATEGORY'])

# Split the dataset into a training set and a test set
df_train_new, df_test_new = train_test_split(df_new, test_size=0.2, random_state=42)

# Check the sizes of the training set and the test set
df_train_new.shape, df_test_new.shape

# Define the formula for the mixed-effects model
formula_new = target_new + ' ~ ' + ' + '.join(features_new)

# Define the groups for the mixed-effects model
groups_new = df_train_new['CATEGORY']

# Fit the mixed-effects model
mixed_model_new = smf.mixedlm(formula_new, df_train_new, groups=groups_new).fit()

'''
LinAlgError: Singular matrix
The dataset does not contain any missing data. Every cell in the dataset is filled.
Error in the previous step was most likely not due to missing data but likely due to multicollinearity 
which is a high correlation between predictor variables.

It appears that the linear algebra operation failed because of a singular matrix which essentially 
means that the matrix (or more specifically the X'X matrix in the context of linear regression) is 
not invertible. This is usually due to multicollinearity in the data which means there are variables 
in the data that are highly correlated with each other making it impossible for the model to 
distinguish their effects.
There are several ways to address this issue, including:
Removing highly correlated features.
Using regularization techniques that can handle multicollinearity such as Ridge or Lasso regression.
Using Principal Component Analysis (PCA) or other dimensionality reduction methods to reduce the number 
of features.
'''

# Trying to determine the highly correlated features and removing them
# Calculate the correlation matrix of the features
corr_matrix = df_train_new[features_new].corr().abs()
# Select the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
# Drop highly correlated features
df_train_new = df_train_new.drop(to_drop, axis=1)
df_test_new = df_test_new.drop(to_drop, axis=1)

# Update the features list
features_new = df_train_new.columns.drop([target_new, 'CATEGORY'])
df_train_new.shape, df_test_new.shape, len(features_new)

# Define the formula for the mixed-effects model
formula_new = target_new + ' ~ ' + ' + '.join(features_new)
# Define the groups for the mixed-effects model
groups_new = df_train_new['CATEGORY']

# Fit the mixed-effects model
mixed_model_new = smf.mixedlm(formula_new, df_train_new, groups=groups_new).fit()

'''
We are still getting the below error:
LinAlgError: Singular matrix
'''

# Load the dataset
# dataset with 186 features with CATEGORY (cluster groups) and no dummy variables
df_new_2 = pd.read_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered_cleaned_rename_delete_union_of_intersections_CATEGORY.csv')

# Define the target variable and the features
target = 'TARGET_Weighted_RetailPrice'
# all features less the target variable and grouping variables
features = df_new_2.columns.drop([target, 'PROFILE_Make', 'CATEGORY'])

# Split the dataset into a training set and a test set
df_train_new_2, df_test_new_2 = train_test_split(df_new_2, test_size=0.2, random_state=42)

# Check the sizes of the training set and the test set
df_train_new_2.shape, df_test_new_2.shape


# Defining the formula for the mixed-effects model
formula_new_2 = target + ' ~ ' + ' + '.join(features)

# Define the groups for the mixed-effects model
groups_new_2 = df_train_new_2['PROFILE_Make'] + "_" + df_train_new_2['CATEGORY']

# Fit the mixed-effects model
mixed_model_new_2 = smf.mixedlm(formula_new_2, df_train_new_2, groups=groups_new_2).fit()

# Show the model summary
mixed_model_new_2.summary()


# Predictions on the test set using the fitted model
# Predict the target variable for the test and train set
df_test_new_2['TARGET_Weighted_RetailPrice_pred'] = mixed_model_new_2.predict(df_test_new_2)
df_train_new_2['TARGET_Weighted_RetailPrice_pred'] = mixed_model_new_2.predict(df_train_new_2)

# Show the first few rows of the test set
df_test_new_2[['TARGET_Weighted_RetailPrice', 'TARGET_Weighted_RetailPrice_pred']].head()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(df_test_new_2['TARGET_Weighted_RetailPrice'], df_test_new_2['TARGET_Weighted_RetailPrice_pred'])

# Calculate the Mean Squared Error (MSE)
# used to calculate RMSE
mse_test = mean_squared_error(df_test_new_2['TARGET_Weighted_RetailPrice'], df_test_new_2['TARGET_Weighted_RetailPrice_pred'])
mse_train = mean_squared_error(df_train_new_2['TARGET_Weighted_RetailPrice'], df_train_new_2['TARGET_Weighted_RetailPrice_pred'])

# Calculate the Coefficient of Determination (R^2)
r2_test = r2_score(df_test_new_2['TARGET_Weighted_RetailPrice'], df_test_new_2['TARGET_Weighted_RetailPrice_pred'])
r2_train = r2_score(df_train_new_2['TARGET_Weighted_RetailPrice'], df_train_new_2['TARGET_Weighted_RetailPrice_pred'])

# Calculate the Root Mean Squared Error (RMSE)
rmse_test = np.sqrt(mse_test)
rmse_train = np.sqrt(mse_train)

# Now apply this function on your data
# LinAlgError: Singular matrix
# cv_r2 = custom_cross_val_score(formula_new_2, df_train_new_2, groups_new_2)

mae, mse_test, mse_train, r2_test, r2_train, rmse_test, rmse_train #cv_r2

'''
mae, mse_test, mse_train, r2_test, r2_train, rmse_test, rmse_train
results are poor
Out[77]: 
(442631.29877478577,
 201202340148.31406,
 201661752269.12283,
 -490.0910258710559,
 -526.3215993612317,
 448555.83838393417,
 449067.6477649251)
'''

# Load the dataset
# dataset with 94 features with CATEGORY (cluster groups) and no dummy variables
df_new_3 = pd.read_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered_cleaned_rename_delete_intersection_CATEGORY_additional.csv')

# Define the target variable and the features
target = 'TARGET_Weighted_RetailPrice'
# all features less the target variable and grouping variables
features = df_new_3.columns.drop([target, 'PROFILE_Make', 'CATEGORY'])

# Split the dataset into a training set and a test set
df_train_new_3, df_test_new_3 = train_test_split(df_new_3, test_size=0.2, random_state=42)

# Check the sizes of the training set and the test set
df_train_new_3.shape, df_test_new_3.shape


# Defining the formula for the mixed-effects model
formula_new_3 = target + ' ~ ' + ' + '.join(features)

# Define the groups for the mixed-effects model
groups_new_3 = df_train_new_3['PROFILE_Make'] + "_" + df_train_new_3['CATEGORY']

# Fit the mixed-effects model
mixed_model_new_3 = smf.mixedlm(formula_new_3, df_train_new_3, groups=groups_new_3).fit()

# Show the model summary
mixed_model_new_3.summary()


# Predictions on the test set using the fitted model
# Predict the target variable for the test set
df_test_new_3['TARGET_Weighted_RetailPrice_pred'] = mixed_model_new_3.predict(df_test_new_3)
df_train_new_3['TARGET_Weighted_RetailPrice_pred'] = mixed_model_new_3.predict(df_train_new_3)

# Show the first few rows of the test set
df_test_new_3[['TARGET_Weighted_RetailPrice', 'TARGET_Weighted_RetailPrice_pred']].head()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate the MAE, MSE, R2, RMSE
# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(df_test_new_3['TARGET_Weighted_RetailPrice'], df_test_new_3['TARGET_Weighted_RetailPrice_pred'])
# Calculate the Mean Squared Error (MSE)
# used to calculate RMSE
mse_test = mean_squared_error(df_test_new_3['TARGET_Weighted_RetailPrice'], df_test_new_3['TARGET_Weighted_RetailPrice_pred'])
mse_train = mean_squared_error(df_train_new_3['TARGET_Weighted_RetailPrice'], df_train_new_3['TARGET_Weighted_RetailPrice_pred'])
# Calculate the Coefficient of Determination (R^2)
r2_test = r2_score(df_test_new_3['TARGET_Weighted_RetailPrice'], df_test_new_3['TARGET_Weighted_RetailPrice_pred'])
r2_train = r2_score(df_train_new_3['TARGET_Weighted_RetailPrice'], df_train_new_3['TARGET_Weighted_RetailPrice_pred'])
# Calculate the Root Mean Squared Error (RMSE)
rmse_test = np.sqrt(mse_test)
rmse_train = np.sqrt(mse_train)
        
# Now apply this function on your data
cv_r2 = custom_cross_val_score(formula_new_3, df_train_new_3, groups_new_3)
print(f"CV R^2: {cv_r2}")

mae, mse_test, mse_train, r2_test, r2_train, rmse_test, rmse_train, cv_r2

'''
Out[47]: 
(4452.07351468766,
 41493434.48758512,
 39370761.75776872,
 0.8987235770001437,
 0.8970501206868893,
 6441.539760615091,
 6274.612478692905,
 0.8921971294967243)
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

# Adding features one at a time to the dataset with 94 features + CATEGORY (cluster groups) and no dummy variables
# Load the datasets
df_A = pd.read_csv(csv_file_2 +'df_for_modeling_all_macro_final_filtered_cleaned_rename_delete_intersection_CATEGORY.csv')
df_B = pd.read_csv(csv_file_2 +'df_for_modeling_all_macro_final_filtered_cleaned_rename_delete_union_of_intersections_CATEGORY.csv')

# Identify columns in B not in A
cols_B_not_in_A = df_B.columns.difference(df_A.columns)

# Define the target variable
target = 'TARGET_Weighted_RetailPrice'

# Initialize a dataframe to store the metrics
metrics_df = pd.DataFrame(columns=['Model', 'Column', 'MAE', 'MSE_train', 'MSE_test', 'R2_train', 'R2_test', 'RMSE_train', 'RMSE_test', 'CV_R2'])

# Loop over the columns of B
for i, col in enumerate(cols_B_not_in_A):
    try:
        # Add the column to A
        df_A[col] = df_B[col]

        # Define the features
        features = df_A.columns.drop([target, 'PROFILE_Make', 'CATEGORY'])

        # Split the dataset into a training set and a test set
        df_train, df_test = train_test_split(df_A, test_size=0.2, random_state=42)

        # Define the formula for the mixed-effects model
        formula = target + ' ~ ' + ' + '.join(features)

        # Define the groups for the mixed-effects model
        groups = df_train['PROFILE_Make'] + "_" + df_train['CATEGORY']

        # Fit the mixed-effects model
        mixed_model = smf.mixedlm(formula, df_train, groups=groups).fit()
        cv_r2 = custom_cross_val_score(formula, df_train, groups)
        
        # Save the model to disk
        pickle.dump(mixed_model, open(f"mixed_model_{i}.pkl", "wb"))
        
        # Predict the target variable for the test set
        df_test.loc[:, 'TARGET_Weighted_RetailPrice_pred'] = mixed_model.predict(df_test)
        df_train.loc[:, 'TARGET_Weighted_RetailPrice_pred'] = mixed_model.predict(df_train)

        # Calculate the MAE, MSE, R2, RMSE
        mae = mean_absolute_error(df_test['TARGET_Weighted_RetailPrice'], df_test['TARGET_Weighted_RetailPrice_pred'])
        mse_test = mean_squared_error(df_test['TARGET_Weighted_RetailPrice'], df_test['TARGET_Weighted_RetailPrice_pred'])
        mse_train = mean_squared_error(df_train['TARGET_Weighted_RetailPrice'], df_train['TARGET_Weighted_RetailPrice_pred'])
        r2_test = r2_score(df_test['TARGET_Weighted_RetailPrice'], df_test['TARGET_Weighted_RetailPrice_pred'])
        r2_train = r2_score(df_train['TARGET_Weighted_RetailPrice'], df_train['TARGET_Weighted_RetailPrice_pred'])
        rmse_test = np.sqrt(mse_test)
        rmse_train = np.sqrt(mse_train)

        # Add the metrics to the dataframe
        metrics_df = metrics_df.append({'Model': f"mixed_model_{i}.pkl", 'Column': col, 'MAE': mae, 'MSE_train': mse_train, 
                                        'MSE_test': mse_test, 'R2_train': r2_train, 'R2_test': r2_test, 
                                        'RMSE_train': rmse_train, 'RMSE_test': rmse_test, 'CV_R2': cv_r2}, 
                                        ignore_index=True)

    except:
        # If error in fitting the model, remove the column from A and skip to the next column
        df_A = df_A.drop(columns=[col])
        continue

# Save the DataFrame to a CSV file
# metrics_df.to_csv(csv_file_2 + 'metrics_df.csv', index=False)
# Load DataFrame
# metrics_df = pd.read_csv(csv_file_2 + 'metrics_df.csv')

'''
mixed_model_34.pkl
SPEC_FRPGseat_lumbar_electric - 3rd to last column
MAE	           MSE_train	 MSE_test	    R2_train	  R2_test	      RMSE_train	RMSE_test	  CV_R2
4420.868842	   38019566.45	 40316456.87	0.900583336	  0.901596323	  6166.000847	6349.524145	  0.89523878
'''

# dropping last two columns in df_A
df_A_new = df_A.iloc[:, :-2]

# Save the DataFrame to a CSV file
# df_A_new.to_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered_cleaned_rename_delete_intersection_CATEGORY_additional_final.csv', index=False)
# Load DataFrame
# df_A_new = pd.read_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered_cleaned_rename_delete_intersection_CATEGORY_additional_final.csv')

# Import the necessary libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the dataset
# dataset with 112 features with CATEGORY (cluster groups) and no dummy variables
df_new_4 = pd.read_csv(csv_file_2 + 'df_for_modeling_all_macro_final_filtered_cleaned_rename_delete_intersection_CATEGORY_additional_final.csv')

# df_new_4 = df_new_3.copy()
# Define the target variable and the features
target = 'TARGET_Weighted_RetailPrice'
# all features less the target variable and grouping variables
features = df_new_4.columns.drop([target, 'PROFILE_Make', 'CATEGORY'])

# Split the dataset into a training set and a test set
df_train_new_4, df_test_new_4 = train_test_split(df_new_4, test_size=0.2, random_state=42)

# Check the sizes of the training set and the test set
df_train_new_4.shape, df_test_new_4.shape


# Defining the formula for the mixed-effects model
formula_new_4 = target + ' ~ ' + ' + '.join(features)

# Define the groups for the mixed-effects model
groups_new_4 = df_train_new_4['PROFILE_Make'] + "_" + df_train_new_4['CATEGORY']

# Fit the mixed-effects model
mixed_model_new_4 = smf.mixedlm(formula_new_4, df_train_new_4, groups=groups_new_4).fit()

# Show the model summary
mixed_model_new_4.summary()

# saving model summary in dataframe
summary_df = mixed_model_new_4.summary().tables[1]
# Convert non-numeric strings to NaN, then to float
summary_df['P>|z|'] = pd.to_numeric(summary_df['P>|z|'], errors='coerce')
# Find significant variables with their p-values
significant_vars = [(idx, p_val) for idx, p_val in zip(summary_df[summary_df['P>|z|'] < 0.05].index, summary_df[summary_df['P>|z|'] < 0.05]['P>|z|'].values)]
# Find non-significant variables with their p-values
non_significant_vars = [(idx, p_val) for idx, p_val in zip(summary_df[summary_df['P>|z|'] >= 0.05].index, summary_df[summary_df['P>|z|'] >= 0.05]['P>|z|'].values)]
# Number of significant variables
num_significant_vars = len(significant_vars)
print(f'Number of significant variables: {num_significant_vars}')


# Predictions on the test set using the fitted model
# Predict the target variable for the train and test set
df_test_new_4['TARGET_Weighted_RetailPrice_pred'] = mixed_model_new_4.predict(df_test_new_4)
df_train_new_4['TARGET_Weighted_RetailPrice_pred'] = mixed_model_new_4.predict(df_train_new_4)

# Show the first few rows of the test set
df_test_new_4[['TARGET_Weighted_RetailPrice', 'TARGET_Weighted_RetailPrice_pred']].head()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate the MAE, MSE, R2, RMSE
# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(df_test_new_4['TARGET_Weighted_RetailPrice'], df_test_new_4['TARGET_Weighted_RetailPrice_pred'])
# Calculate the Mean Squared Error (MSE)
# used to calculate RMSE
mse_test = mean_squared_error(df_test_new_4['TARGET_Weighted_RetailPrice'], df_test_new_4['TARGET_Weighted_RetailPrice_pred'])
mse_train = mean_squared_error(df_train_new_4['TARGET_Weighted_RetailPrice'], df_train_new_4['TARGET_Weighted_RetailPrice_pred'])
# Calculate the Coefficient of Determination (R^2)
r2_test = r2_score(df_test_new_4['TARGET_Weighted_RetailPrice'], df_test_new_4['TARGET_Weighted_RetailPrice_pred'])
r2_train = r2_score(df_train_new_4['TARGET_Weighted_RetailPrice'], df_train_new_4['TARGET_Weighted_RetailPrice_pred'])
# Calculate the Root Mean Squared Error (RMSE)
rmse_test = np.sqrt(mse_test)
rmse_train = np.sqrt(mse_train)
        
# Now apply this function on your data
cv_r2 = custom_cross_val_score(formula_new_4, df_train_new_4, groups_new_4)

mae, mse_test, mse_train, r2_test, r2_train, rmse_test, rmse_train, cv_r2

'''
Out[95]: 
(4420.868842282509,
 40316456.8738271,
 38019566.44693405,
 0.9015963226319371,
 0.9005833364025192,
 6349.524145463745,
 6166.000847140232,
 0.8952387801922093)
'''