import pandas as pd
import numpy as np

# -------------------------------
# 1. Load the dataset
# -------------------------------
input_path = 'collected.csv'
df = pd.read_csv(input_path)

# -------------------------------
# 2. Omit rows with many missing features until we have 1000 entries
# -------------------------------
df['missing_count'] = df.isnull().sum(axis=1)
df_sorted = df.sort_values('missing_count')
cleaned_df = df_sorted.head(1000).copy()
cleaned_df.drop(columns=['missing_count'], inplace=True)

# -------------------------------
# 3. Drop unwanted features (columns) such as 'title' and 'address'
# -------------------------------
cols_to_drop = ['title', 'address']
cleaned_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# -------------------------------
# 4. Handle missing values without omitting data
# -------------------------------
for col in cleaned_df.columns:
    if cleaned_df[col].dtype == 'object':  
        # Categorical column: Fill missing values with the most frequent value (mode)
        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    else:
        # Numerical column: Fill missing values with the median
        cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)

# -------------------------------
# 5. Convert 'area' column to square feet without omitting the data
# -------------------------------
def convert_to_sqft(area):
    if isinstance(area, str):
        area = area.lower().strip()
        if 'sq.m' in area:
            num = float(area.replace('sq.m', '').strip())
            return num * 10.7639  # Convert to square feet
        elif 'aana' in area:
            num = float(area.replace('aana', '').strip())
            return num * 342.25
        elif 'ropani' in area:
            num = float(area.replace('ropani', '').strip())
            return num * 5476
        elif 'dhur' in area:
            num = float(area.replace('dhur', '').strip())
            return num * 182.25
        elif 'sq.ft' in area:
            return float(area.replace('sq.ft', '').strip())
    return area  # Return original value if conversion fails

if 'area' in cleaned_df.columns:
    cleaned_df['area_sqft'] = cleaned_df['area'].apply(convert_to_sqft)
    cleaned_df.drop(columns=['area'], inplace=True)

# -------------------------------
# 6. One-hot encode the 'city' column (kathmandu, lalitpur, bhaktapur)
# -------------------------------
if 'city' in cleaned_df.columns:
    cleaned_df['city'] = cleaned_df['city'].astype(str).str.lower()
    allowed_cities = ['kathmandu', 'lalitpur', 'bhaktapur']
    cleaned_df['city'] = cleaned_df['city'].apply(lambda x: x if x in allowed_cities else 'other')
    
    # One-hot encode city values
    cleaned_df['kathmandu'] = (cleaned_df['city'] == 'kathmandu').astype(int)
    cleaned_df['lalitpur'] = (cleaned_df['city'] == 'lalitpur').astype(int)
    cleaned_df['bhaktapur'] = (cleaned_df['city'] == 'bhaktapur').astype(int)
    cleaned_df.drop(columns=['city'], inplace=True)  # Drop the original city column

# -------------------------------
# 7. Binary encoding of amenities (each with a separate column)
# -------------------------------
amenity_cols = ['parking', 'water supply', 'frontyard', 'backyard', 'lawn']
for col in amenity_cols:
    if col in cleaned_df.columns:
        cleaned_df[col] = cleaned_df[col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', 'true', '1'] else 0)

# -------------------------------
# 8. Convert all columns to numeric values (without omitting the data)
# -------------------------------
for col in cleaned_df.columns:
    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')

# -------------------------------
# 9. Save the cleaned dataset to a new CSV file
# -------------------------------
output_path = 'cleaned.csv'
cleaned_df.to_csv(output_path, index=False)
