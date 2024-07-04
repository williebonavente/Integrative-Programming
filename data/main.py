import pandas as pd
import random

# Define the popular chord progressions
popular_progressions = [
    ['C', 'G', 'Am', 'F'],  # I-V-vi-IV
    ['C', 'Am', 'F', 'G'],  # I-vi-IV-V
    ['C', 'G', 'F', 'G']    # I-V-VI-V
]

# All possible chords, simplified for example purposes
all_chords = ['A', 'Am', 'B', 'Bm', 'C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm']

# Create a list to hold all the progressions, ensuring popular ones are more frequent
all_progressions = popular_progressions * 10  # Add each popular progression 10 times

# Generate additional random progressions to reach a total of 70
while len(all_progressions) < 70:
    random_progression = random.sample(all_chords, 4)  # Randomly select 4 chords for the progression
    all_progressions.append(random_progression)

# Shuffle the list to mix popular and random progressions
random.shuffle(all_progressions)

# Create a DataFrame where each chord progression is in its own row
df_popular_progressions = pd.DataFrame(all_progressions)

# Export to Excel, each chord progression in its own row
df_popular_progressions.to_excel("randomized_chord_progressions.xlsx", index=False, header=False)