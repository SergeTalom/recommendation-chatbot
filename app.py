import streamlit as st
from recommender_logic import (
    smart_recommend_from_prompt,
    recommend_next_step_from_history,
    AVAILABLE_TOPICS,
    AVAILABLE_LEVELS,
    AVAILABLE_TYPES,
)

st.set_page_config(page_title="Smart Course Recommender", layout="wide")

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
                    f"   Type: {row['resource_type']}\n"
                    f"   Cluster: {row['sent_clusters']}\n"
                    f"   Match confidence: {row['recommendation_confidence']}%\n\n"
                )

    st.session_state.chat_history.append(("Bot", bot_reply))

# display conversation
for speaker, message in st.session_state.chat_history:
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
                f"Type: {item['resource_type']}"
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
        selected_type = st.selectbox(
            "Preferred resource type",
            options=AVAILABLE_TYPES,
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
                    f"   Type: {row['resource_type']}\n"
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
            f"({recs_df.iloc[idx]['learner_level']}, {recs_df.iloc[idx]['resource_type']}, "
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