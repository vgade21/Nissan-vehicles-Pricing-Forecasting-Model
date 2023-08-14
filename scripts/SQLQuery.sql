-- Check row count of each spec table
SELECT COUNT(*) FROM [dbo].[Spec_combined_1];
SELECT COUNT(*) FROM [dbo].[Spec_combined_2];
SELECT COUNT(*) FROM [dbo].[Spec_combined_3];

-- Check column count of each spec table
SELECT COUNT(COLUMN_NAME) 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE 
    TABLE_CATALOG = 'Capstone' 
    AND TABLE_SCHEMA = 'dbo' 
    AND TABLE_NAME = 'Spec_combined_3';

-- Check column count of each spec table
SELECT TOP 100 *
FROM [dbo].[Spec_combined_1];

-- Create a new table to hold the horizontally joined data
-- The problem is the max number of columns in a sql server table is 1024
SELECT *
INTO Spec_combined
FROM (
    -- Select data from Table_A with index
    SELECT ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Index_A, *
    FROM Spec_combined_1
) AS A
FULL OUTER JOIN (
    -- Select data from Table_B with index
    SELECT ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Index_B, *
    FROM Spec_combined_2
) AS B ON A.Index_A = B.Index_B
FULL OUTER JOIN (
    -- Select data from Table_C with index
    SELECT ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Index_C, *
    FROM Spec_combined_3
) AS C ON A.Index_A = C.Index_C;
