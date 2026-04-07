import pandas as pd
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "resources.csv"
CLUSTER_COL = "sent_clusters"
TOPIC_CONFIDENCE_THRESHOLD = 0.25
LEVEL_CONFIDENCE_THRESHOLD = 0.40
TYPE_CONFIDENCE_THRESHOLD = 0.45

level_order = {
    "beginner": 0,
    "intermediate": 1,
    "advanced": 2
}

resources_df = pd.read_csv(DATA_PATH).copy()

resources_df["normalized_topic"] = resources_df["relevant_topic"].astype(str).str.strip().str.lower()
resources_df["normalized_level"] = resources_df["learner_level"].astype(str).str.strip().str.lower()
resources_df["normalized_type"] = resources_df["resource_type"].astype(str).str.strip().str.lower()

resources_df["combined_text"] = (
    resources_df["title"].fillna("").astype(str) + " " +
    resources_df["description"].fillna("").astype(str)
)

resources_df["level_rank"] = resources_df["normalized_level"].map(level_order)
AVAILABLE_TOPICS = sorted(resources_df["normalized_topic"].dropna().unique().tolist())
AVAILABLE_LEVELS = sorted(resources_df["normalized_level"].dropna().unique().tolist())
AVAILABLE_TYPES = sorted(resources_df["normalized_type"].dropna().unique().tolist())

# similarity vectorizer
similarity_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
similarity_vectorizer.fit(resources_df["combined_text"])


clf_texts = resources_df["combined_text"].tolist()

def build_classifier(labels):
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=3000))
    ])
    clf.fit(clf_texts, labels)
    return clf

topic_clf = build_classifier(resources_df["relevant_topic"])
level_clf = build_classifier(resources_df["learner_level"])
type_clf = build_classifier(resources_df["resource_type"])

def predict_with_confidence(clf, text):
    probs = clf.predict_proba([text])[0]
    best_idx = probs.argmax()
    pred = clf.classes_[best_idx]
    conf = float(probs[best_idx])
    return pred, conf

def understand_prompt(user_prompt):
    topic, topic_conf = predict_with_confidence(topic_clf, user_prompt)
    level, level_conf = predict_with_confidence(level_clf, user_prompt)
    rtype, type_conf = predict_with_confidence(type_clf, user_prompt)

    return {
        "topic": topic,
        "topic_conf": topic_conf,
        "level": level,
        "level_conf": level_conf,
        "resource_type": rtype,
        "resource_type_conf": type_conf
    }

def smart_recommend_from_prompt(user_prompt, top_k=5, overrides=None):
    intent = understand_prompt(user_prompt)
    overrides = overrides or {}

    topic = str(overrides.get("topic", intent["topic"])).lower()
    level = str(overrides.get("level", intent["level"])).lower()
    rtype = str(overrides.get("resource_type", intent["resource_type"])).lower()
    needs_clarification = []

    if intent["topic_conf"] < TOPIC_CONFIDENCE_THRESHOLD and "topic" not in overrides:
        needs_clarification.append("topic")
    if intent["level_conf"] < LEVEL_CONFIDENCE_THRESHOLD and "level" not in overrides:
        needs_clarification.append("level")
    if intent["resource_type_conf"] < TYPE_CONFIDENCE_THRESHOLD and "resource_type" not in overrides:
        needs_clarification.append("resource_type")

    candidates = resources_df[
        resources_df["normalized_topic"] == str(topic).lower()
    ].copy()

    if candidates.empty:
        return intent, candidates, needs_clarification

    if level in AVAILABLE_LEVELS:
        candidates = candidates[candidates["normalized_level"] == level].copy()
    if rtype in AVAILABLE_TYPES:
        candidates = candidates[candidates["normalized_type"] == rtype].copy()

    if candidates.empty:
        candidates = resources_df[
            resources_df["normalized_topic"] == str(topic).lower()
        ].copy()

    prompt_vec = similarity_vectorizer.transform([str(user_prompt)])
    cand_vec = similarity_vectorizer.transform(candidates["combined_text"].astype(str))

    sim_scores = cosine_similarity(prompt_vec, cand_vec).flatten()

    candidates["final_score"] = sim_scores
    candidates["recommendation_confidence"] = (candidates["final_score"] * 100).clip(0, 100).round(2)

    recs = (
        candidates
        .sort_values("final_score", ascending=False)
        .head(top_k)
        .copy()
    )

    return intent, recs, needs_clarification

def recommend_next_step_from_history(user_history, top_k=3):
    if not user_history:
        return pd.DataFrame()

    recent = user_history[-3:]

    cluster_counts = Counter([
        x.get(CLUSTER_COL, x.get("cluster"))
        for x in recent
        if x.get(CLUSTER_COL, x.get("cluster")) is not None
    ])
    if not cluster_counts:
        return pd.DataFrame()
    target_cluster = cluster_counts.most_common(1)[0][0]

    max_level = max(
        level_order.get(x["learner_level"].lower(), 0)
        for x in recent
    )

    next_level = min(max_level + 1, 2)

    seen_titles = {x["title"] for x in user_history}

    candidates = resources_df[
        (resources_df[CLUSTER_COL] == target_cluster) &
        (resources_df["level_rank"] >= next_level) &
        (~resources_df["title"].isin(seen_titles))
    ].copy()

    return candidates.head(top_k)

    