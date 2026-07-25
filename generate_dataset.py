"""
generate_dataset.py

Creates controlled synthetic behavioural data
for AI Nudge Stage 1 MVP.
"""

import pandas as pd
import random


rows = []


activities = [
    "Studying",
    "Working",
    "Exercise",
    "Phone",
    "Relaxing"
]


weather = [
    "Sunny",
    "Cloudy",
    "Rain"
]


social = [
    "Low",
    "Medium",
    "High"
]


for i in range(5000):


    sleep = round(random.uniform(3,10),1)

    stress = random.randint(1,5)

    mood = random.randint(1,5)

    energy = random.randint(1,5)

    screen_time = round(random.uniform(0,12),1)

    activity = random.choice(activities)

    weather_value = random.choice(weather)

    social_value = random.choice(social)

    hour = random.randint(0,23)



    # -----------------------------
    # Clear expert rules
    # -----------------------------


    if sleep < 5 and hour >= 21:

        nudge = "Prepare for bed"


    elif screen_time > 7:

        nudge = "Reduce screen time"


    elif stress >= 4 and energy <= 2:

        nudge = "Take a short break"


    elif mood <= 2 and social_value == "Low":

        nudge = "Connect socially"


    elif activity == "Exercise":

        nudge = "Stay active"


    elif (
        sleep >= 7
        and stress <=2
        and mood >=4
        and energy >=4
    ):

        nudge = "Maintain habits"


    elif energy <=2:

        nudge = "Take a short break"


    else:

        nudge = random.choice(
            [
                "Drink water",
                "Stretch",
                "Go outside"
            ]
        )



    rows.append({

        "sleep": sleep,

        "stress": stress,

        "mood": mood,

        "energy": energy,

        "screen_time": screen_time,

        "activity": activity,

        "weather": weather_value,

        "social": social_value,

        "hour": hour,

        "best_nudge": nudge

    })



df = pd.DataFrame(rows)


df.to_csv(
    "data/training.csv",
    index=False
)


print("Generated 5000 training examples.")
print("Saved to data/training.csv")