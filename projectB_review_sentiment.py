# Project B: Restaurant Review Sentiment vs Rating Mismatch
# Dataset: Google Maps Restaurant Reviews - kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews
# Replace the synthetic reviews list below with pd.read_csv("reviews.csv")[["text","rating"]] to use the real file.

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

reviews = [
    ("The food was absolutely amazing, best meal I've had all year!", 5),
    ("Terrible service, waited an hour and the food was cold.", 1),
    ("It was okay, nothing special but not bad either.", 3),
    ("Loved the ambiance and the staff were super friendly.", 5),
    ("Overpriced for what you get. Would not come back.", 2),
    ("Honestly disappointed, the reviews online were misleading.", 1),
    ("Great place! Fast service and tasty food.", 4),
    ("Not the worst but definitely not great either, mediocre experience.", 3),
    ("I was blown away by the quality and presentation.", 5),
    ("Rude staff ruined what could have been a decent meal.", 2),
    ("Average food, average service, nothing memorable.", 3),
    ("Best restaurant in town, highly recommend to everyone!", 5),
    ("Food was fine but I had to send it back once.", 1),
    ("Pleasant surprise, exceeded my expectations completely.", 5),
    ("Waited too long and the order was wrong when it arrived.", 2),
    ("Solid food, friendly staff, will come back again.", 4),
    ("Nothing went wrong here, it was a fine experience overall.", 1),
    ("Absolutely wonderful evening, the chef outdid himself.", 5),
    ("Food poisoning symptoms the next day, never again.", 1),
    ("Decent value for the price, satisfied with the visit.", 4),
]

df = pd.DataFrame(reviews, columns=["review_text", "rating"])

analyzer = SentimentIntensityAnalyzer()
df["sentiment_score"] = df["review_text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])

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

df["sentiment_label"] = df["sentiment_score"].apply(sentiment_label)
df["rating_label"] = df["rating"].apply(rating_label)
df["mismatch"] = df["sentiment_label"] != df["rating_label"]

print("Full results:\n")
print(df[["review_text", "rating", "sentiment_score", "sentiment_label", "rating_label", "mismatch"]]
      .to_string(index=False))

mismatches = df[df["mismatch"]]
print(f"\n{len(mismatches)} of {len(df)} reviews have a sentiment/rating mismatch "
      f"({len(mismatches)/len(df)*100:.0f}%)")
print("\nFlagged reviews (possible fake/manipulated ratings or sarcasm):")
for _, row in mismatches.iterrows():
    print(f"  Rating={row['rating']} but sentiment='{row['sentiment_label']}': \"{row['review_text']}\"")

print(f"\nAverage sentiment by star rating:")
print(df.groupby("rating")["sentiment_score"].mean().round(3))

df.to_csv("projectB_output_data.csv", index=False)
print("\nSaved full results to projectB_output_data.csv")
