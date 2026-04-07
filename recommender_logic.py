import pandas as pd
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
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

def build_classifier(labels, estimator):
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", estimator)
    ])
    clf.fit(clf_texts, labels)
    return clf

MODEL_FAMILY_BUILDERS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=3000),
    "SGDClassifier": lambda: SGDClassifier(loss="log_loss", max_iter=2000, random_state=42),
    "MultinomialNB": lambda: MultinomialNB(alpha=0.5),
}


def build_model_family(labels):
    return {
        model_name: build_classifier(labels, estimator_builder())
        for model_name, estimator_builder in MODEL_FAMILY_BUILDERS.items()
    }


topic_models = build_model_family(resources_df["relevant_topic"])
level_models = build_model_family(resources_df["learner_level"])
type_models = build_model_family(resources_df["resource_type"])

def predict_with_confidence(clf, text):
    probs = clf.predict_proba([text])[0]
    best_idx = probs.argmax()
    pred = clf.classes_[best_idx]
    conf = float(probs[best_idx])
    return pred, conf


def predict_with_model_family(model_family, text):
    model_scores = {}
    best_model = None
    best_prediction = None
    best_conf = -1.0

    for model_name, clf in model_family.items():
        pred, conf = predict_with_confidence(clf, text)
        model_scores[model_name] = {
            "prediction": pred,
            "confidence": conf,
        }
        if conf > best_conf:
            best_conf = conf
            best_prediction = pred
            best_model = model_name

    return best_prediction, best_conf, best_model, model_scores

def understand_prompt(user_prompt):
    topic, topic_conf, topic_model, topic_scores = predict_with_model_family(
        topic_models,
        user_prompt
    )
    level, level_conf, level_model, level_scores = predict_with_model_family(
        level_models,
        user_prompt
    )
    rtype, type_conf, type_model, type_scores = predict_with_model_family(
        type_models,
        user_prompt
    )

    return {
        "topic": topic,
        "topic_conf": topic_conf,
        "topic_model": topic_model,
        "topic_model_scores": topic_scores,
        "level": level,
        "level_conf": level_conf,
        "level_model": level_model,
        "level_model_scores": level_scores,
        "resource_type": rtype,
        "resource_type_conf": type_conf,
        "resource_type_model": type_model,
        "resource_type_model_scores": type_scores,
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

    topic_candidates = resources_df[
        resources_df["normalized_topic"] == str(topic).lower()
    ].copy()

    if topic_candidates.empty:
        return intent, topic_candidates, needs_clarification

    candidate_slices = []
    if level in AVAILABLE_LEVELS and rtype in AVAILABLE_TYPES:
        strict_slice = topic_candidates[
            (topic_candidates["normalized_level"] == level) &
            (topic_candidates["normalized_type"] == rtype)
        ].copy()
        if not strict_slice.empty:
            candidate_slices.append(strict_slice)

    if level in AVAILABLE_LEVELS:
        level_slice = topic_candidates[
            topic_candidates["normalized_level"] == level
        ].copy()
        if not level_slice.empty:
            candidate_slices.append(level_slice)

    if rtype in AVAILABLE_TYPES:
        type_slice = topic_candidates[
            topic_candidates["normalized_type"] == rtype
        ].copy()
        if not type_slice.empty:
            candidate_slices.append(type_slice)

    candidate_slices.append(topic_candidates)

    candidates = (
        pd.concat(candidate_slices, ignore_index=False)
        .drop_duplicates(subset=["title"], keep="first")
        .copy()
    )

    prompt_vec = similarity_vectorizer.transform([str(user_prompt)])
    cand_vec = similarity_vectorizer.transform(candidates["combined_text"].astype(str))

    sim_scores = cosine_similarity(prompt_vec, cand_vec).flatten()

    candidates["final_score"] = sim_scores
    candidates["level_match"] = (candidates["normalized_level"] == level).astype(float)
    candidates["type_match"] = (candidates["normalized_type"] == rtype).astype(float)
    # preference-first ranking: exact user preferences are prioritized before pure similarity
    candidates["preference_tier"] = (
        (2 * candidates["level_match"]) +
        (2 * candidates["type_match"])
    )
    candidates["refined_score"] = (
        candidates["final_score"] +
        (0.02 * candidates["level_match"]) +
        (0.02 * candidates["type_match"])
    )
    candidates["recommendation_confidence"] = (candidates["refined_score"] * 100).clip(0, 100).round(2)

    ranked = candidates.sort_values(
        ["preference_tier", "refined_score"],
        ascending=[False, False]
    ).copy()

    diversified = (
        ranked
        .groupby(CLUSTER_COL, as_index=False, sort=False)
        .head(1)
        .sort_values(["preference_tier", "refined_score"], ascending=[False, False])
        .copy()
    )

    if len(diversified) < top_k:
        already_titles = set(diversified["title"].tolist())
        filler = ranked[~ranked["title"].isin(already_titles)].copy()
        needed = top_k - len(diversified)
        diversified = pd.concat([diversified, filler.head(needed)], ignore_index=False)

    recs = diversified.head(top_k).copy()

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

    return (
        candidates
        .drop_duplicates(subset=["title"], keep="first")
        .head(top_k)
        .copy()
    )

    