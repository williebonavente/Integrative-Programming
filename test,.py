import pandas as pd

# Path to your Excel file
excel_file_path = 'opm_old_songs_tagalog_oldies_playlist.xlsx'

# Load the Excel file and select the second sheet
df = pd.read_excel(excel_file_path, sheet_name=1, header=None)  # sheet_name=1 for the second sheet

# Function to extract the root note from a chord name
def extract_root_note(chord):
    # Check if the chord is a string
    if not isinstance(chord, str):
        return None  # Return None for non-string values
    if pd.isnull(chord):
        return None  # Return None for empty cells
    if len(chord) > 1 and chord[1] in ['#', 'b']:  # Ensure the chord has at least 2 characters
        return chord[:2]  # Include the sharp or flat in the root note
    else:
        return chord[0]  # Return only the first character if no sharp or flat
# Apply the function to each chord in the DataFrame
root_notes_df = df.applymap(extract_root_note)

# Now you have a DataFrame of root notes, you can process them as needed
# Specify the path to the new Excel file where you want to save the root notes
output_excel_file_path = 'output_root_notes.xlsx'

# Write the DataFrame to the new Excel file
root_notes_df.to_excel(output_excel_file_path, index=False)

print(f'Root notes DataFrame has been written to {output_excel_file_path}')

print(root_notes_df)