# Review Sentiment Checker — the user end of Project B.
# A user pastes a restaurant review, picks the star rating that came with it,
# and instantly sees whether the words and the stars agree.
# Logic is identical to projectB_review_sentiment.py (VADER, +/-0.3 thresholds).

import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def sentiment_label(score):
    if score >= 0.3:
        return "positive"
    elif score <= -0.3:
        return "negative"
    return "neutral"


def rating_label(r):
    if r >= 4:
        return "positive"
    elif r <= 2:
        return "negative"
    return "neutral"


EXAMPLES = {
    "Pick one…": ("", 5),
    "A happy 5-star review": ("The food was absolutely amazing, best meal I've had all year!", 5),
    "A 1-star review that reads happy": ("Nothing went wrong here, it was a fine experience overall.", 1),
    "A 5-star review the tool can't read": ("I was blown away by the quality and presentation.", 5),
    "An angry 1-star review": ("Terrible service, waited an hour and the food was cold.", 1),
}

st.set_page_config(page_title="Review Sentiment Checker", page_icon="⭐")
st.title("⭐ Review Sentiment Checker")
st.write(
    "Paste a restaurant review and the star rating it was given. "
    "The checker reads the **words** and tells you if they match the **stars**."
)

example = st.selectbox("Try an example (optional)", list(EXAMPLES.keys()))
ex_text, ex_stars = EXAMPLES[example]

review_text = st.text_area("Review text", value=ex_text, height=120,
                           placeholder="Type or paste the review here…")
stars = st.select_slider("Star rating given with this review", options=[1, 2, 3, 4, 5],
                         value=ex_stars)

if st.button("Check this review"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        score = analyzer.polarity_scores(review_text)["compound"]
        text_says = sentiment_label(score)
        stars_say = rating_label(stars)

        col1, col2, col3 = st.columns(3)
        col1.metric("Text sentiment score", f"{score:+.3f}")
        col2.metric("The words say", text_says)
        col3.metric("The stars say", stars_say)

        if text_says == stars_say:
            st.success(f"✅ MATCH — the words and the {stars}-star rating agree ({text_says}).")
        else:
            st.error(
                f"⚠️ MISMATCH — the words read {text_says} but the rating is {stars} star(s) "
                f"({stars_say}). This review would be flagged for a second look."
            )
        st.caption("How it works: VADER scores the text from −1 to +1. "
                   "Scores ≥ +0.3 are positive, ≤ −0.3 negative. "
                   "Stars 4–5 are positive, 1–2 negative, 3 neutral. Different labels = mismatch.")
