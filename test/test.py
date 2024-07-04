import pandas as pd

# Define the notes
notes = ['A', 'B', 'C', 'D', 'E', 'F', 'F#', 'G']

# Generate all possible transitions from one note to another
transitions = [f"{note1} -> {note2}" for note1 in notes for note2 in notes if note1 != note2]

# Convert the transitions into a DataFrame where each transition is a separate column
transitions_df = pd.DataFrame([transitions])

# Transpose the DataFrame to get all transitions in a single row
# Note: This step is not necessary as we already structured the DataFrame to have a single row

# Write the DataFrame to an Excel file
transitions_df.to_excel('note_transitions_single_row.xlsx', index=False, header=False)

print('Transitions have been written to note_transitions_single_row.xlsx')