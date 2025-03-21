import pandas as pd
import os

# Set the folder where all the files are stored
folder_path = "path/to/your/folder"  # Change this to your actual folder path
output_file = "combined_output.xlsx"  # Change as needed

# List all files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith(('.csv', '.xlsx'))]

# Initialize an empty list to store dataframes
df_list = []

for file in files:
    file_path = os.path.join(folder_path, file)
    
    # Read the file (CSV or Excel)
    if file.endswith('.csv'):
        df = pd.read_csv(file_path, header=None)  # No header assumed
    else:
        df = pd.read_excel(file_path, header=None)  # No header assumed
    
    # Ensure only 5 columns are taken
    df = df.iloc[:, :5]  
    
    df_list.append(df)  # Append dataframe to the list

# Concatenate all dataframes
combined_df = pd.concat(df_list, ignore_index=True)

# Save to an Excel file
combined_df.to_excel(output_file, index=False, header=False)

print(f"All files combined and saved to {output_file}")
