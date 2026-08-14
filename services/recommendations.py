"""
services/recommendations.py

NudgeWise v2.7
Contextual Action Engine.

PURPOSE
-------
Translate the six high-level NudgeWise AI intervention classes into
specific, practical suggestions based on the participant's current
check-in.

Architecture:

    v2.6 AI probability model
            ↓
    intervention class
            ↓
    v2.7 contextual action engine
            ↓
    specific recommendation
            ↓
    explanation
            ↓
    participant feedback

IMPORTANT
---------
This module does NOT override the trained AI prediction.

It only personalises the action shown within the intervention category
selected by the v2.6 model.

The actions are low-risk wellbeing suggestions and are not medical,
clinical, or therapeutic instructions.

Exact action-selection scores are product design rules, not published
clinical effect sizes.
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# Recommendation object
# ============================================================

@dataclass(frozen=True)
class ContextualRecommendation:
    """
    User-facing recommendation produced after the AI has selected
    a high-level intervention category.
    """

    intervention: str

    title: str

    action: str

    reason: str

    action_id: str


# ============================================================
# Helpers
# ============================================================

def _safe_float(
    value,
    default: float = 0.0,
) -> float:

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_int(
    value,
    default: int = 0,
) -> int:

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# Maintain habits
# ============================================================

def _maintain_habits(
    sleep: float,
    stress: int,
    mood: int,
    energy: int,
    screen_time: float,
    activity_minutes: int,
    connectedness: int,
) -> ContextualRecommendation:
    """
    Identify the strongest currently stable behaviour rather than
    giving the vague instruction "Maintain habits".
    """

    candidates = []


    # --------------------------------------------------------
    # Sleep
    # --------------------------------------------------------

    if 8.0 <= sleep <= 10.0:

        candidates.append(
            (
                3.0,
                ContextualRecommendation(
                    intervention="Maintain habits",
                    title="Keep your sleep routine going",
                    action=(
                        "Your reported sleep is currently within the "
                        "adolescent reference range used by NudgeWise. "
                        "Try to keep a similar sleep opportunity tonight."
                    ),
                    reason=(
                        "Sleep was one of the stronger stable parts "
                        "of this check-in."
                    ),
                    action_id="maintain_sleep",
                ),
            )
        )


    # --------------------------------------------------------
    # Physical activity
    # --------------------------------------------------------

    if activity_minutes >= 60:

        candidates.append(
            (
                2.5,
                ContextualRecommendation(
                    intervention="Maintain habits",
                    title="Keep your activity routine going",
                    action=(
                        "You've already reported a solid amount of "
                        "physical activity today. Keep the routine "
                        "that helped you get moving."
                    ),
                    reason=(
                        "Your reported activity was already relatively strong."
                    ),
                    action_id="maintain_activity",
                ),
            )
        )


    # --------------------------------------------------------
    # Connectedness
    # --------------------------------------------------------

    if connectedness >= 4:

        candidates.append(
            (
                2.2,
                ContextualRecommendation(
                    intervention="Maintain habits",
                    title="Keep making time for connection",
                    action=(
                        "Your connectedness rating is relatively strong "
                        "today. Keep making space for the people and "
                        "interactions that are helping."
                    ),
                    reason=(
                        "Connectedness was one of the stronger parts "
                        "of this check-in."
                    ),
                    action_id="maintain_connection",
                ),
            )
        )


    # --------------------------------------------------------
    # Current wellbeing
    # --------------------------------------------------------

    if (
        stress <= 2
        and mood >= 4
        and energy >= 4
    ):

        candidates.append(
            (
                2.8,
                ContextualRecommendation(
                    intervention="Maintain habits",
                    title="Keep today's balance going",
                    action=(
                        "Your current stress, mood and energy are all "
                        "tracking relatively well. Keep doing what is "
                        "working rather than forcing another change."
                    ),
                    reason=(
                        "Your current wellbeing ratings did not show "
                        "a strong need for another intervention."
                    ),
                    action_id="maintain_balance",
                ),
            )
        )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not candidates:

        return ContextualRecommendation(
            intervention="Maintain habits",
            title="Keep your current routine steady",
            action=(
                "No single area stood out strongly enough to justify "
                "a major change right now. Keep your routine steady "
                "and check in again later."
            ),
            reason=(
                "The AI did not identify a stronger competing "
                "intervention from this check-in."
            ),
            action_id="maintain_general",
        )


    candidates.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )


    return candidates[
        0
    ][1]


# ============================================================
# Take a short break
# ============================================================

def _short_break(
    stress: int,
    energy: int,
    activity: str,
    hour: int,
) -> ContextualRecommendation:

    if activity in {
        "Studying",
        "Working",
    }:

        return ContextualRecommendation(
            intervention="Take a short break",
            title="Take a short reset from your task",
            action=(
                "Step away from your current task for a few minutes. "
                "When you return, choose one clear next task instead "
                "of trying to restart everything at once."
            ),
            reason=(
                "You were in a cognitively demanding context, which "
                "made a short recovery break more relevant."
            ),
            action_id="break_cognitive",
        )


    if activity == "Phone":

        return ContextualRecommendation(
            intervention="Take a short break",
            title="Take a brief break from your phone",
            action=(
                "Put your phone aside for a short period and switch "
                "to something offline before deciding what to do next."
            ),
            reason=(
                "Your current phone context made an offline pause "
                "more immediately actionable."
            ),
            action_id="break_phone",
        )


    if energy <= 2:

        return ContextualRecommendation(
            intervention="Take a short break",
            title="Give yourself a low-effort pause",
            action=(
                "Take a short pause before starting something else. "
                "Choose something easy and restorative rather than "
                "adding another demanding task immediately."
            ),
            reason=(
                "Your reported energy was relatively low."
            ),
            action_id="break_low_energy",
        )


    if stress >= 4:

        return ContextualRecommendation(
            intervention="Take a short break",
            title="Step away for a short reset",
            action=(
                "Pause what you're doing for a few minutes and return "
                "once you've had some distance from the immediate task."
            ),
            reason=(
                "Your reported stress was relatively high."
            ),
            action_id="break_stress",
        )


    if hour >= 21:

        return ContextualRecommendation(
            intervention="Take a short break",
            title="Slow things down for a moment",
            action=(
                "Take a short pause before starting anything new. "
                "Because it's later in the day, keep the break calm "
                "rather than turning it into another long activity."
            ),
            reason=(
                "The time of day made a lower-intensity pause more suitable."
            ),
            action_id="break_evening",
        )


    return ContextualRecommendation(
        intervention="Take a short break",
        title="Take a short reset",
        action=(
            "Step away briefly from what you're doing, then return "
            "with one clear next step."
        ),
        reason=(
            "The combined pattern in your check-in supported "
            "a short recovery pause."
        ),
        action_id="break_general",
    )


# ============================================================
# Prepare for bed
# ============================================================

def _prepare_for_bed(
    sleep: float,
    screen_time: float,
    activity: str,
    hour: int,
) -> ContextualRecommendation:

    if activity == "Phone":

        return ContextualRecommendation(
            intervention="Prepare for bed",
            title="Start winding down from your phone",
            action=(
                "When you're ready to finish what you're doing, move "
                "away from recreational screen use and start your "
                "usual bedtime routine."
            ),
            reason=(
                "The model combined your sleep and time-of-day context "
                "with your current phone use."
            ),
            action_id="bed_phone",
        )


    if screen_time >= 6.0:

        return ContextualRecommendation(
            intervention="Prepare for bed",
            title="Shift into a quieter evening routine",
            action=(
                "Start moving away from recreational screens and toward "
                "the routine you normally use before sleep."
            ),
            reason=(
                "Your reported screen exposure and evening context "
                "both contributed to this recommendation."
            ),
            action_id="bed_screen",
        )


    if sleep < 7.0:

        return ContextualRecommendation(
            intervention="Prepare for bed",
            title="Protect your sleep opportunity tonight",
            action=(
                "Try not to push your bedtime later unnecessarily. "
                "Give yourself enough time for a full night's sleep."
            ),
            reason=(
                "Your reported sleep was below the adolescent reference "
                "range used by NudgeWise."
            ),
            action_id="bed_short_sleep",
        )


    return ContextualRecommendation(
        intervention="Prepare for bed",
        title="Start your wind-down routine",
        action=(
            "Begin shifting from active tasks into the routine you "
            "normally use to prepare for sleep."
        ),
        reason=(
            "The combination of sleep and time-of-day inputs made "
            "bedtime preparation the strongest intervention."
        ),
        action_id="bed_general",
    )


# ============================================================
# Reduce screen time
# ============================================================

def _reduce_screen_time(
    screen_time: float,
    activity: str,
    hour: int,
) -> ContextualRecommendation:

    if activity == "Phone":

        return ContextualRecommendation(
            intervention="Reduce screen time",
            title="Make your next activity screen-free",
            action=(
                "When you finish what you're currently doing, choose "
                "one offline activity before returning to recreational "
                "screen use."
            ),
            reason=(
                "Your recreational screen exposure and current phone "
                "context both made an offline switch more relevant."
            ),
            action_id="screen_phone",
        )


    if hour >= 20:

        return ContextualRecommendation(
            intervention="Reduce screen time",
            title="Create a screen-free part of your evening",
            action=(
                "Choose a period of your evening to spend away from "
                "recreational screens and switch to something offline."
            ),
            reason=(
                "Screen exposure was relevant and the check-in occurred "
                "later in the day."
            ),
            action_id="screen_evening",
        )


    if screen_time >= 6.0:

        return ContextualRecommendation(
            intervention="Reduce screen time",
            title="Break up your recreational screen time",
            action=(
                "Take a meaningful break from recreational screens "
                "before your next session rather than continuing "
                "straight through."
            ),
            reason=(
                "Your reported recreational screen exposure was one "
                "of the stronger inputs supporting this intervention."
            ),
            action_id="screen_high_exposure",
        )


    return ContextualRecommendation(
        intervention="Reduce screen time",
        title="Take an offline break",
        action=(
            "Step away from recreational screens for a while and "
            "choose an offline activity before returning."
        ),
        reason=(
            "The combined screen and context pattern increased the "
            "model's preference for reducing screen exposure."
        ),
        action_id="screen_general",
    )


# ============================================================
# Stay active
# ============================================================

def _stay_active(
    activity_minutes: int,
    activity: str,
    energy: int,
) -> ContextualRecommendation:

    if activity == "Exercise":

        return ContextualRecommendation(
            intervention="Stay active",
            title="Keep your current activity going",
            action=(
                "You're already being active right now. Continue your "
                "current activity rather than adding another exercise task."
            ),
            reason=(
                "Your current context shows that you're already active."
            ),
            action_id="active_currently_exercising",
        )


    if energy <= 2:

        return ContextualRecommendation(
            intervention="Stay active",
            title="Choose some light movement",
            action=(
                "If it feels appropriate, add some easy movement rather "
                "than forcing an intense workout."
            ),
            reason=(
                "Your activity exposure was relatively low, but your "
                "energy was also low, so the suggestion is intentionally gentle."
            ),
            action_id="active_low_energy",
        )


    if activity_minutes < 30:

        return ContextualRecommendation(
            intervention="Stay active",
            title="Add some movement to your day",
            action=(
                "Look for an opportunity to add some moderate movement "
                "today, such as walking, sport, cycling or another "
                "activity you enjoy."
            ),
            reason=(
                "Your reported activity minutes were relatively low."
            ),
            action_id="active_low_minutes",
        )


    return ContextualRecommendation(
        intervention="Stay active",
        title="Add a little more movement",
        action=(
            "Find a convenient opportunity for some movement today "
            "that fits around what you're already doing."
        ),
        reason=(
            "Your activity pattern made movement the strongest "
            "current intervention."
        ),
        action_id="active_general",
    )


# ============================================================
# Connect socially
# ============================================================

def _connect_socially(
    connectedness: int,
    mood: int,
    activity: str,
) -> ContextualRecommendation:

    if connectedness <= 2:

        return ContextualRecommendation(
            intervention="Connect socially",
            title="Reach out to someone you trust",
            action=(
                "Consider messaging, calling or spending some time "
                "with someone you feel comfortable talking to."
            ),
            reason=(
                "Your connectedness rating was relatively low today."
            ),
            action_id="social_low_connection",
        )


    if activity == "Phone":

        return ContextualRecommendation(
            intervention="Connect socially",
            title="Turn screen time into real connection",
            action=(
                "If you're already on your phone, consider using some "
                "of that time to talk directly with someone rather than "
                "only scrolling or browsing."
            ),
            reason=(
                "Your current phone context made direct social contact "
                "an immediately available option."
            ),
            action_id="social_phone",
        )


    if mood <= 2:

        return ContextualRecommendation(
            intervention="Connect socially",
            title="Spend some time with someone supportive",
            action=(
                "Consider spending a little time with someone you "
                "trust or feel comfortable around."
            ),
            reason=(
                "Your connectedness pattern and current mood together "
                "increased the model's preference for social connection."
            ),
            action_id="social_low_mood",
        )


    return ContextualRecommendation(
        intervention="Connect socially",
        title="Make space for a meaningful connection",
        action=(
            "Look for an opportunity to talk or spend time with "
            "someone you value today."
        ),
        reason=(
            "Your check-in pattern made social connection the strongest "
            "current intervention."
        ),
        action_id="social_general",
    )


# ============================================================
# Public recommendation interface
# ============================================================

def personalise_recommendation(
    prediction: str,
    sleep,
    stress,
    mood,
    energy,
    screen_time,
    activity_minutes,
    connectedness,
    activity,
    hour,
) -> ContextualRecommendation:
    """
    Translate an AI intervention class into a specific action.

    This function NEVER changes the intervention selected by the AI.
    """

    sleep = _safe_float(
        sleep,
        7.5,
    )

    stress = _safe_int(
        stress,
        3,
    )

    mood = _safe_int(
        mood,
        3,
    )

    energy = _safe_int(
        energy,
        3,
    )

    screen_time = _safe_float(
        screen_time,
        0.0,
    )

    activity_minutes = _safe_int(
        activity_minutes,
        0,
    )

    connectedness = _safe_int(
        connectedness,
        3,
    )

    hour = _safe_int(
        hour,
        12,
    )


    if prediction == "Maintain habits":

        return _maintain_habits(
            sleep=sleep,
            stress=stress,
            mood=mood,
            energy=energy,
            screen_time=screen_time,
            activity_minutes=activity_minutes,
            connectedness=connectedness,
        )


    if prediction == "Take a short break":

        return _short_break(
            stress=stress,
            energy=energy,
            activity=activity,
            hour=hour,
        )


    if prediction == "Prepare for bed":

        return _prepare_for_bed(
            sleep=sleep,
            screen_time=screen_time,
            activity=activity,
            hour=hour,
        )


    if prediction == "Reduce screen time":

        return _reduce_screen_time(
            screen_time=screen_time,
            activity=activity,
            hour=hour,
        )


    if prediction == "Stay active":

        return _stay_active(
            activity_minutes=activity_minutes,
            activity=activity,
            energy=energy,
        )


    if prediction == "Connect socially":

        return _connect_socially(
            connectedness=connectedness,
            mood=mood,
            activity=activity,
        )


    # Unknown model class should never normally occur.

    return ContextualRecommendation(
        intervention=prediction,
        title=prediction,
        action=(
            "Use this recommendation as a prompt to reflect on "
            "what would be most helpful right now."
        ),
        reason=(
            "NudgeWise could not generate a more specific "
            "contextual action for this intervention."
        ),
        action_id="fallback",
    )


# ============================================================
# Manual tests
# ============================================================

if __name__ == "__main__":

    scenarios = [
        {
            "prediction":
                "Take a short break",

            "sleep":
                7.0,

            "stress":
                5,

            "mood":
                3,

            "energy":
                2,

            "screen_time":
                4.0,

            "activity_minutes":
                20,

            "connectedness":
                3,

            "activity":
                "Studying",

            "hour":
                16,
        },

        {
            "prediction":
                "Maintain habits",

            "sleep":
                8.5,

            "stress":
                2,

            "mood":
                4,

            "energy":
                4,

            "screen_time":
                3.0,

            "activity_minutes":
                75,

            "connectedness":
                4,

            "activity":
                "Relaxing",

            "hour":
                18,
        },

        {
            "prediction":
                "Connect socially",

            "sleep":
                8.0,

            "stress":
                3,

            "mood":
                2,

            "energy":
                3,

            "screen_time":
                4.0,

            "activity_minutes":
                40,

            "connectedness":
                1,

            "activity":
                "Phone",

            "hour":
                19,
        },
    ]


    print()
    print("=" * 72)
    print(
        "NUDGEWISE V2.7 CONTEXTUAL ACTION ENGINE"
    )
    print("=" * 72)


    for scenario in scenarios:

        result = personalise_recommendation(
            **scenario
        )


        print()
        print(
            f"AI intervention: "
            f"{result.intervention}"
        )

        print(
            f"Title: "
            f"{result.title}"
        )

        print(
            f"Action: "
            f"{result.action}"
        )

        print(
            f"Reason: "
            f"{result.reason}"
        )

        print(
            f"Action ID: "
            f"{result.action_id}"
        )

        print("-" * 72)