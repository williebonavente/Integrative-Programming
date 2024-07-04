import pandas as pd

# Load the Excel file, ensuring the first row is used as the header
excel_file_path = 'output_root_notes.xlsx'
df = pd.read_excel(excel_file_path, header=0)  # Adjust the header parameter if needed

# Function to process each row and return a Series indicating note frequency
def process_row(row):
    # Collect notes from all relevant columns (assuming they start from the second column)
    notes = [note for note in row[1:] if not pd.isnull(note)]
    # Count the occurrences of each note
    note_frequency = pd.Series({note: notes.count(note) for note in notes})
    return note_frequency

# Skip the first row of actual data, then apply the function to each row
note_frequency_df = df.iloc[1:].apply(process_row, axis=1).fillna(0)

# Convert to integer
note_frequency_df = note_frequency_df.astype(int)

# Save the resulting DataFrame to a new Excel file
output_excel_file_path = 'processed_root_notes_frequency.xlsx'
note_frequency_df.to_excel(output_excel_file_path, index=False)

print(f'Processed DataFrame has been written to {output_excel_file_path}')