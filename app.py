import pandas as pd
import streamlit as st
from recommender_logic import (
    smart_recommend_from_prompt,
    recommend_next_step_from_history,
    AVAILABLE_TOPICS,
    AVAILABLE_LEVELS,
    AVAILABLE_TYPES,
)

st.set_page_config(page_title="Smart Course Recommender", layout="wide")

RESOURCE_TYPE_LABELS = {
    "youtube_playlist": "YouTube Video",
    "youtube_video": "YouTube Video",
    "video": "Video",
    "book": "Book",
    "course": "Course",
    "article": "Article",
    "podcast": "Podcast",
    "documentation": "Documentation",
}


def friendly_resource_type(value):
    raw = str(value).strip().lower()
    if raw in RESOURCE_TYPE_LABELS:
        return RESOURCE_TYPE_LABELS[raw]
    return raw.replace("_", " ").title()


def percent(value):
    return f"{float(value) * 100:.1f}%"


def friendly_text(value):
    return str(value).strip().replace("_", " ").title()


def confidence_for(model_scores, model_name):
    details = model_scores.get(model_name)
    if not details:
        return "-"
    return f"{float(details['confidence']) * 100:.1f}%"


def build_intent_transparency_table(intent):
    rows = []
    config = [
        (
            "Topic",
            "topic",
            "topic_extracted",
            "topic_model_prediction",
            "topic_conf",
            "topic_source",
            "topic_reason",
            "topic_model_scores",
            friendly_text
        ),
        (
            "Level",
            "level",
            "level_extracted",
            "level_model_prediction",
            "level_conf",
            "level_source",
            "level_reason",
            "level_model_scores",
            friendly_text
        ),
        (
            "Resource type",
            "resource_type",
            "resource_type_extracted",
            "resource_type_model_prediction",
            "resource_type_conf",
            "resource_type_source",
            "resource_type_reason",
            "resource_type_model_scores",
            friendly_resource_type,
        ),
    ]

    for (
        label,
        final_key,
        extracted_key,
        model_pred_key,
        conf_key,
        source_key,
        reason_key,
        scores_key,
        formatter
    ) in config:
        model_scores = intent.get(scores_key, {}) or {}
        model_pred = formatter(intent.get(model_pred_key, "N/A"))
        extracted_value = intent.get(extracted_key)
        extracted_display = "-" if extracted_value is None else formatter(extracted_value)
        top_conf = float(intent.get(conf_key, 0.0))
        source_used = intent.get(source_key, "N/A")
        rows.append(
            {
                "Target": label,
                "Extracted from prompt": extracted_display,
                "Model prediction": model_pred,
                "Model confidence": f"{top_conf * 100:.1f}%",
                "Final selected": formatter(intent.get(final_key, "N/A")),
                "Source used": source_used,
                "Reason": intent.get(reason_key, "N/A"),
                "LogisticRegression": confidence_for(model_scores, "LogisticRegression"),
                "SGDClassifier": confidence_for(model_scores, "SGDClassifier"),
                "MultinomialNB": confidence_for(model_scores, "MultinomialNB"),
                "UserSelection": confidence_for(model_scores, "UserSelection"),
            }
        )

    return pd.DataFrame(rows)


st.title("🎓 Smart Course Recommender Chatbot")
st.caption("You can always ask a new learning goal in the chat box, even after making a selection.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_history" not in st.session_state:
    st.session_state.user_history = []

if "last_recs" not in st.session_state:
    st.session_state.last_recs = None

if "awaiting_selection" not in st.session_state:
    st.session_state.awaiting_selection = False

if "recs_version" not in st.session_state:
    st.session_state.recs_version = 0

if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

if "clarify_fields" not in st.session_state:
    st.session_state.clarify_fields = []

user_prompt = st.chat_input("Describe what you want to learn, or ask a new follow-up...")

if user_prompt:
    st.session_state.chat_history.append(("User", user_prompt))

    intent, recs, needs_clarification = smart_recommend_from_prompt(user_prompt)

    if needs_clarification:
        st.session_state.awaiting_clarification = True
        st.session_state.awaiting_selection = False
        st.session_state.pending_prompt = user_prompt
        st.session_state.clarify_fields = needs_clarification
        st.session_state.last_recs = None

        needs_text = ", ".join(needs_clarification)
        bot_reply = (
            "I need a bit more detail before recommending. "
            f"My confidence is low for: {needs_text}. "
            "Here is what I predicted so far:\n\n"
            f"- Topic: {friendly_text(intent['topic'])} ({percent(intent['topic_conf'])})\n"
            f"- Level: {friendly_text(intent['level'])} ({percent(intent['level_conf'])})\n"
            f"- Resource type: {friendly_resource_type(intent['resource_type'])} ({percent(intent['resource_type_conf'])})\n\n"
            "Please choose your preferences below."
        )
    else:
        st.session_state.awaiting_clarification = False
        st.session_state.last_recs = recs
        st.session_state.awaiting_selection = not recs.empty
        st.session_state.recs_version += 1

        if recs.empty:
            bot_reply = "Sorry, I could not find recommendations."
        else:
            bot_reply = (
                "Here are my recommendations with confidence scores. "
                "Please choose one below:\n\n"
            )
            for i, (_, row) in enumerate(recs.iterrows(), start=1):
                bot_reply += (
                    f"{i}. {row['title']}\n"
                    f"   Topic: {row['relevant_topic']}\n"
                    f"   Level: {row['learner_level']}\n"
                    f"   Type: {friendly_resource_type(row['resource_type'])}\n"
                    f"   Cluster: {row['sent_clusters']}\n"
                    f"   Match confidence: {row['recommendation_confidence']}%\n\n"
                )

    st.session_state.chat_history.append(
        {
            "speaker": "Bot",
            "message": bot_reply,
            "intent": intent,
        }
    )

# display conversation
for entry in st.session_state.chat_history:
    if isinstance(entry, dict):
        speaker = entry.get("speaker", "Bot")
        message = entry.get("message", "")
        with st.chat_message("user" if speaker == "User" else "assistant"):
            st.write(message)
            if speaker == "Bot" and entry.get("intent") is not None:
                st.caption("Prediction transparency")
                st.table(build_intent_transparency_table(entry["intent"]))
    else:
        speaker, message = entry
        with st.chat_message("user" if speaker == "User" else "assistant"):
            st.write(message)

with st.sidebar:
    st.subheader("Your learning path so far")
    if st.session_state.user_history:
        for idx, item in enumerate(st.session_state.user_history, start=1):
            st.markdown(
                f"{idx}. **{item['title']}**  \n"
                f"Topic: {item['relevant_topic']} | "
                f"Level: {item['learner_level']} | "
                f"Type: {friendly_resource_type(item['resource_type'])}"
            )
    else:
        st.write("No course selected yet.")

if st.session_state.awaiting_clarification and st.session_state.pending_prompt:
    st.subheader("Help me refine your request")

    overrides = {}
    if "topic" in st.session_state.clarify_fields:
        selected_topic = st.selectbox(
            "Preferred topic",
            options=AVAILABLE_TOPICS,
            key="clarify_topic",
        )
        overrides["topic"] = selected_topic
    if "level" in st.session_state.clarify_fields:
        selected_level = st.selectbox(
            "Preferred level",
            options=AVAILABLE_LEVELS,
            key="clarify_level",
        )
        overrides["level"] = selected_level
    if "resource_type" in st.session_state.clarify_fields:
        type_options = AVAILABLE_TYPES
        selected_type = st.selectbox(
            "Preferred resource type",
            options=type_options,
            format_func=friendly_resource_type,
            key="clarify_type",
        )
        overrides["resource_type"] = selected_type

    if st.button("Apply preferences", key="apply_clarification"):
        prefs_text = ", ".join([f"{k}={v}" for k, v in overrides.items()])
        st.session_state.chat_history.append(("User", f"My preferences: {prefs_text}"))
        intent, recs, _ = smart_recommend_from_prompt(
            st.session_state.pending_prompt,
            overrides=overrides
        )

        # Keep one transparency table per original prompt:
        # update the latest bot entry intent instead of creating a new one.
        for idx in range(len(st.session_state.chat_history) - 1, -1, -1):
            entry = st.session_state.chat_history[idx]
            if isinstance(entry, dict) and entry.get("speaker") == "Bot" and entry.get("intent") is not None:
                st.session_state.chat_history[idx]["intent"] = intent
                break

        st.session_state.awaiting_clarification = False
        st.session_state.last_recs = recs
        st.session_state.awaiting_selection = not recs.empty
        st.session_state.recs_version += 1
        st.session_state.pending_prompt = ""
        st.session_state.clarify_fields = []

        if recs.empty:
            bot_reply = "Thanks. I still could not find recommendations with those preferences."
        else:
            bot_reply = "Great, here are refined recommendations:\n\n"
            for i, (_, row) in enumerate(recs.iterrows(), start=1):
                bot_reply += (
                    f"{i}. {row['title']}\n"
                    f"   Topic: {row['relevant_topic']}\n"
                    f"   Level: {row['learner_level']}\n"
                    f"   Type: {friendly_resource_type(row['resource_type'])}\n"
                    f"   Cluster: {row['sent_clusters']}\n"
                    f"   Match confidence: {row['recommendation_confidence']}%\n\n"
                )
        st.session_state.chat_history.append(("Bot", bot_reply))
        st.rerun()

if (
    st.session_state.awaiting_selection
    and st.session_state.last_recs is not None
    and not st.session_state.last_recs.empty
):
    st.subheader("Choose one recommendation to continue")

    recs_df = st.session_state.last_recs.reset_index(drop=True)
    choice_key = f"rec_choice_{st.session_state.recs_version}"

    selected_idx = st.radio(
        "Recommended options",
        options=list(range(len(recs_df))),
        format_func=lambda idx: (
            f"{idx + 1}. {recs_df.iloc[idx]['title']} "
            f"({recs_df.iloc[idx]['learner_level']}, {friendly_resource_type(recs_df.iloc[idx]['resource_type'])}, "
            f"cluster {recs_df.iloc[idx]['sent_clusters']})"
        ),
        key=choice_key,
    )

    if st.button("Confirm selection", key=f"confirm_choice_{st.session_state.recs_version}"):
        selected_row = recs_df.iloc[selected_idx]
        selected_title = selected_row["title"]

        st.session_state.chat_history.append(
            ("User", f"I choose: {selected_title}")
        )

        history_entry = {
            "title": selected_row["title"],
            "description": selected_row.get("description", ""),
            "relevant_topic": selected_row["relevant_topic"],
            "learner_level": selected_row["learner_level"],
            "resource_type": selected_row["resource_type"],
            "sent_clusters": selected_row["sent_clusters"],
        }
        st.session_state.user_history.append(history_entry)

        followup = recommend_next_step_from_history(st.session_state.user_history)

        if not followup.empty:
            followup_text = (
                f"Nice choice. Since you selected '{selected_title}', "
                "here is a suggested next learning step:\n\n"
            )
            for j, (_, r) in enumerate(followup.iterrows(), start=1):
                followup_text += (
                    f"{j}. {r['title']}\n"
                    f"   Topic: {r['relevant_topic']}\n"
                    f"   Level: {r['learner_level']}\n\n"
                )
            followup_text += "You can now select another recommendation or type a new learning request in chat."
        else:
            followup_text = (
                f"Nice choice: '{selected_title}'. I do not have another next-step "
                "recommendation yet from your current history. "
                "You can ask for a new topic in the chat box."
            )

        st.session_state.chat_history.append(("Bot", followup_text))
        st.session_state.awaiting_selection = False
        st.session_state.last_recs = None
        st.rerun()