"""
generate_data.py

Creates a synthetic behavioural dataset
for training the AI Nudge model.
"""


import pandas as pd
import random
from pathlib import Path


# -----------------------------
# Settings
# -----------------------------

NUMBER_OF_SAMPLES = 10000


# -----------------------------
# Possible values
# -----------------------------

activities = [
    "Studying",
    "Phone",
    "Relaxing",
    "Working",
    "Exercise"
]


weather_conditions = [
    "Sunny",
    "Cloudy",
    "Rain"
]


nudges = [
    "Drink water",
    "Take a short break",
    "Prepare for bed",
    "Go outside",
    "Stretch",
    "Reduce screen time",
    "Take a walk",
    "Sleep earlier"
]


# -----------------------------
# Generate one example
# -----------------------------

def generate_example():

    sleep = round(random.uniform(3, 10), 1)

    stress = random.randint(1, 5)

    mood = random.randint(1, 5)

    energy = random.randint(1, 5)

    screen_time = round(random.uniform(0, 12), 1)

    activity = random.choice(activities)

    weather = random.choice(weather_conditions)

    hour = random.randint(0, 23)


    # Behaviour rules

    if sleep < 5 and hour >= 21:

        nudge = "Prepare for bed"


    elif screen_time > 7:

        nudge = "Reduce screen time"


    elif energy <= 2:

        nudge = "Take a short break"


    elif stress >= 4:

        nudge = "Take a walk"


    elif weather == "Sunny" and hour < 17:

        nudge = "Go outside"


    elif activity == "Studying":

        nudge = "Stretch"


    else:

        nudge = random.choice(nudges)



    return [

        sleep,
        stress,
        mood,
        energy,
        screen_time,
        activity,
        weather,
        hour,
        nudge

    ]


# -----------------------------
# Create dataset
# -----------------------------

rows = []


for i in range(NUMBER_OF_SAMPLES):

    rows.append(
        generate_example()
    )


columns = [

    "sleep",
    "stress",
    "mood",
    "energy",
    "screen_time",
    "activity",
    "weather",
    "hour",
    "best_nudge"

]


df = pd.DataFrame(
    rows,
    columns=columns
)


# -----------------------------
# Save dataset
# -----------------------------

Path("data").mkdir(exist_ok=True)


df.to_csv(
    "data/training.csv",
    index=False
)


print(
    "Dataset generated successfully."
)


print(
    f"Created {len(df)} examples."
)


print(df.head())