"""
predict.py

Loads the trained AI model,
generates personalised nudges,
and explains recommendations.
"""

import pandas as pd
import joblib


MODEL_PATH = "models/nudge_model.pkl"


# Load model
model = joblib.load(MODEL_PATH)



def predict_nudge(
    sleep,
    stress,
    mood,
    energy,
    screen_time,
    activity,
    day_type,
    social,
    hour
):

    user_data = pd.DataFrame({

        "sleep": [sleep],
        "stress": [stress],
        "mood": [mood],
        "energy": [energy],
        "screen_time": [screen_time],
        "activity": [activity],
        "day_type": [day_type],
        "social": [social],
        "hour": [hour]

    })


    prediction = model.predict(user_data)[0]


    probabilities = model.predict_proba(user_data)[0]

    confidence = max(probabilities)


    return prediction, confidence





def explain_prediction(
    prediction,
    sleep,
    stress,
    screen_time,
    hour,
    energy,
    mood
):

    reasons = []


    if screen_time > 7:
        reasons.append(
            "High screen time detected"
        )


    if sleep < 5:
        reasons.append(
            "Low sleep detected"
        )


    if stress >= 4:
        reasons.append(
            "High stress detected"
        )


    if energy <= 2:
        reasons.append(
            "Low energy detected"
        )


    if mood <= 2:
        reasons.append(
            "Low mood detected"
        )


    if hour >= 21:
        reasons.append(
            "Late time of day"
        )


    # If no negative factors exist,
    # explain based on recommendation

    if len(reasons) == 0:

        if prediction == "Stay active":

            reasons.append(
                "Your activity levels suggest maintaining physical movement"
            )


        elif prediction == "Maintain habits":

            reasons.append(
                "Your habits currently look balanced"
            )


        elif prediction == "Take a short break":

            reasons.append(
                "A short recovery period may improve your energy"
            )


        else:

            reasons.append(
                "Your current habits suggest a small improvement opportunity"
            )


    return reasons





# Test

if __name__ == "__main__":


    result, confidence = predict_nudge(

        sleep=4,

        stress=5,

        mood=2,

        energy=2,

        screen_time=10,

        activity="Studying",

        day_type="School Day",

        social="Low",

        hour=22

    )


    reasons = explain_prediction(

        result,

        sleep=4,

        stress=5,

        screen_time=10,

        hour=22,

        energy=2,

        mood=2

    )


    print("\nAI Recommendation:")
    print(result)


    print(
        f"\nConfidence: {confidence:.0%}"
    )


    print("\nWhy:")

    for reason in reasons:
        print("-", reason)