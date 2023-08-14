# -*- coding: utf-8 -*-
"""
Created on Sat May  6 03:00:17 2023

@author: vamshi gadepally
"""
import pandas as pd

# Path to CSV file
csv_file = 'C:\\Users\\vamsh\\Documents\\Uni\\UChicago\\Spring 2023\\Capstone I_IP06\\Project\\Data\\'

# Load Spec_combined
Spec_combined = pd.read_csv(csv_file + 'Spec\\combined\\Spec_combined.csv')
first_columns = ['Name', 'Make', 'Model', 'Version name', 'Model year']
Spec = Spec_combined[first_columns + [col for col in Spec_combined.columns if col not in first_columns]]

# Print the dimensions of the DataFrame
print('Number of rows:', Spec.shape[0])
print("")
print('Number of columns:', Spec.shape[1])

# Load Inventory_Historical_Data
Inventory = pd.read_excel(csv_file + 'Inventory\\Inventory_Historical_Data.xlsx')
first_columns = ['Unique_ID', 'Approx_Make', 'Approx_Model', 'Segment', 'MakeModel']
Inventory = Inventory[first_columns + [col for col in Inventory.columns if col not in first_columns]]

# Load MSRP
MSRP = pd.read_csv(csv_file + 'MSRP\\MSRP.csv')

# Load NCBS
NCBS = pd.read_csv(csv_file + 'NCBS\\NCBS_Aggregated_edit.csv')

# Load PCC
PCC = pd.read_excel(csv_file + 'PCC\\PCC_edit.xlsx')
first_columns = ['Brand Name', 'Model Name']
last_columns = ['Country_PCC', 'Country']
PCC = PCC[first_columns + [col for col in PCC.columns if col not in first_columns]]
other_columns = [col for col in PCC.columns if col not in last_columns]
PCC = PCC.reindex(columns=other_columns + last_columns)

# Load Segment
Segment = pd.read_excel(csv_file + 'Segment\\Segment_edit.xlsx')

# Load Volume
Volume = pd.read_csv(csv_file + 'Volume\\Volume.csv')

# Load Edmunds_combined
Edmunds_combined = pd.read_excel(csv_file + 'Edmunds\\combined\\Edmunds_combined.xlsx')
first_columns = ['UniversalMessageId', 'ConversationId', 'Intel Product', 'Variant']
Edmunds = Edmunds_combined[first_columns + [col for col in Edmunds_combined.columns if col not in first_columns]]

Edmunds_test = Edmunds.copy()

'''
The following files have erroneous data in the 'Intel Product' column.
ConversationStreamDistribution_256b6f8e-0f28-4a41-b624-81526faecd9d_1.xlsx
MY2019.xlsx
MY2020.xlsx
MY2021.xlsx
MY2022.xlsx
MY2023.xlsx

They have irrelvant text after the car model.
Eg: 
    2022 Chevrolet Equinox - What to Expect | Edmunds
    2022 Toyota Tundra | Edmunds
    2022 Mercedes-Benz C-Class Extends Luxury Appeal for Less Dough | Edmunds

6535 / 57301 rows affected.
2019/2020 have less than 20, 2023 around 600 and 2021/2022 have more than a 1000 rows.
Clean/correct data needs to be provided as it's not possible to clean using a script.
For now these rows have been removed from the data.
'''

# Count rows before deletion
count_before = len(Edmunds_test) # 57301

# Delete rows containing '|'
Edmunds_test = Edmunds_test[~Edmunds_test['Intel Product'].str.contains('\|')]

# Count rows after deletion
count_after = len(Edmunds_test) # 50766

# Calculate the number of rows deleted
print("Number of rows deleted:\n", count_before - count_after) # 6535

# Extract year, make, and model
# Note: any rows that don't start with 4 numbers will return nan
Edmunds_test[['Year', 'Make', 'Model_2']] = Edmunds_test['Intel Product'].str.extract(r'^(\d{4}) (\w+)?(.*)')

# Extract decimal value next to character 'L'
Edmunds_test['Decimal'] = Edmunds_test['Variant'].str.extract(r'(\d+\.\d+)L')

# Rearrange columns
first_columns = ['UniversalMessageId', 'ConversationId', 'Intel Product', 'Year', 'Make', 'Model_2', 'Variant', 'Decimal']
Edmunds_test = Edmunds_test[first_columns + [col for col in Edmunds_test.columns if col not in first_columns]]

'''
=LEN(G2) - LEN(SUBSTITUTE(G2, ".", "")) # check if . is in cell
=IF(ISNUMBER(FIND("*",F2)),"Contains *","") # find * in the cell
=IF(A1="",0,LEN(TRIM(A1))-LEN(SUBSTITUTE(A1," ",""))+1) # count number of words
'''

# Load RV
RV = pd.read_excel(csv_file + 'RV\\combined\\RV_combined.xlsx')
first_columns = ['Make', 'Model', 'Model#', 'ModelDesc', 'ModelYear']
RV = RV[first_columns + [col for col in RV.columns if col not in first_columns]]

RV_test = RV.copy()
RV_test['Decimal'] = RV_test['ModelDesc'].str.extract(r'(\d+\.\d+)')
RV_test['Decimal'].fillna('', inplace=True)
RV_test['Model_2'] = RV_test['Model'].str.replace(r'\s?(AWD|FWD|RWD|4WD|2WD)', '')

first_columns = ['Make', 'Model', 'Model_2', 'Model#', 'ModelDesc', 'Decimal', 'ModelYear']
RV_test = RV_test[first_columns + [col for col in RV_test.columns if col not in first_columns]]


# Load OaO
OaO = pd.read_excel(csv_file + 'OaO\\combined\\OaO_combined.xlsx')
OaO_test = OaO.copy
'''
'Model' column has extra information:
    Altima Hybrid
    Regal Sportback
    Prius Plug-In Hybrid
    B-class	Electric Drive

Some extra information is useful.
Need to figure out how to remove only where required.
'''
import re

# Removing following words from 'Model' column
words_to_remove = [
    'Hybrid', 'Sportback', 'Pickup', 'Rover', 'Cruz', 'Cross', 'Sport', 'Recharge', 'Prime', 'Plug-In',
    'Electric', '4xe', 'Mach', 'E-tron/Sportback', 'Mid-suv', 'Cruiser', 'Crosstrek', 'Select', 'Unlimited',
    'Alltrack', 'Hatchback', 'Denali', 'Lightning', 'Series', 'Drive', 'Door', '(4-door)', 'Sedan'
]

pattern = r'\s?(' + '|'.join(map(re.escape, words_to_remove)) + r')'
OaO_test['Model_2'] = OaO_test['Model'].str.replace(pattern, '', regex=True)