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


def friendly_text(value):
    return str(value).strip().replace("_", " ").title()


def truncate_description(text, max_chars=320):
    s = str(text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def row_to_context(row):
    return {
        "sent_clusters": row["sent_clusters"],
        "learner_level": row["learner_level"],
        "relevant_topic": row["relevant_topic"],
        "title": row["title"],
    }


def level_stated_by_user(intent):
    if not intent:
        return False
    return intent.get("level_source") in ("PromptExtracted", "UserSelection")


def format_recommendation_body(row, intent=None):
    title = row["title"]
    topic = friendly_text(row["relevant_topic"])
    level = friendly_text(row["learner_level"])
    rtype = friendly_resource_type(row["resource_type"])
    desc = truncate_description(row.get("description", ""))
    if level_stated_by_user(intent):
        lead = (
            f"Here’s one that could be a nice fit: **{title}** — it’s a **{level}** **{rtype}** on **{topic}**."
        )
    else:
        lead = (
            f"Here’s one that could be a nice fit: **{title}** — a **{rtype}** on **{topic}**.\n\n"
            f"*The catalog tags this resource as **{level}** difficulty — you didn’t specify a level, so I’m using that tag to narrow results.*"
        )
    return f"{lead}\n\n{desc}"


def invite_more_text(intent):
    if level_stated_by_user(intent):
        return (
            "\n\nWant a couple more in the same ballpark (same level)? Just say **more** or **yes**. "
            "Or ask me something new anytime."
        )
    return (
        "\n\nWant a few more similar picks? Just say **more** or **yes**. "
        "Or ask me something new anytime."
    )



# First message in chat so the greeting is clearly from the assistant, not part of the page title
WELCOME_MESSAGE = (
    "👋 **Hi - I’m your course assistant.**\n\n"
    "I’m here to help you find something concrete to study - at the level and format that fits you.\n\n"
    "**What would you like to work on?** Just describe it below in your own words."
)


def format_extra_rows(rows_df, intent=None):
    lines = []
    for i, (_, r) in enumerate(rows_df.iterrows(), start=1):
        desc = truncate_description(r.get("description", ""), max_chars=200)
        if level_stated_by_user(intent):
            meta = (
                f"{friendly_resource_type(r['resource_type'])}, "
                f"{friendly_text(r['learner_level'])}"
            )
        else:
            meta = (
                f"{friendly_resource_type(r['resource_type'])} · "
                f"catalog: {friendly_text(r['learner_level'])}"
            )
        lines.append(f"**{i}. {r['title']}** ({meta})\n{desc}")
    return "\n\n".join(lines)


def wants_decline(text):
    t = str(text).strip().lower()
    # Refusals of "more" — checked before wants_more so "nothing more" is not read as yes
    if any(
        p in t
        for p in (
            "nothing more",
            "no more",
            "not any more",
            "dont want more",
            "don't want more",
            "need no more",
            "no need for more",
            "that's enough",
            "thats enough",
        )
    ):
        return True
    keywords = (
        "no thanks",
        "no thank",
        "not now",
        "enough",
        "stop",
        "that's all",
        "thats all",
        "all set",
        "nope",
        "nah",
        "don't",
        "dont need",
    )
    return any(k in t for k in keywords) or t in {"no", "n"}


def wants_more(text):
    t = str(text).strip().lower()
    if wants_decline(t):
        return False
    if len(t) > 80:
        return False
    keywords = (
        "yes",
        "yeah",
        "yep",
        "sure",
        "more",
        "show me",
        "please",
        "extra",
        "another",
        "others",
        "suggestions",
        "ideas",
        "couple",
        "few",
        "2",
        "3",
        "go on",
        "continue",
        "list",
    )
    return any(k in t for k in keywords) or t in {"y", "ok", "okay"}


def looks_like_fresh_topic(text):
    """Long or question-like input → treat as new learning request."""
    t = str(text).strip()
    if len(t) > 100:
        return True
    starters = ("i want", "i'd like", "teach me", "learn", "how ", "what ", "show me how", "find me")
    return any(t.lower().startswith(s) for s in starters) and len(t) > 25


def append_bot_message(text):
    st.session_state.chat_history.append({"speaker": "Bot", "message": text})


def format_preferences_user_message(overrides):
    """Turn clarification picks into a natural chat line (no raw dataset keys)."""
    if not overrides:
        return "Here’s what works for me."
    topic = friendly_text(overrides["topic"]) if "topic" in overrides else None
    level = friendly_text(overrides["level"]) if "level" in overrides else None
    rtype = friendly_resource_type(overrides["resource_type"]) if "resource_type" in overrides else None

    if topic and level and rtype:
        return (
            f"I’d like to study **{topic}** at **{level}** level, "
            f"preferably as **{rtype}**."
        )
    if topic and level:
        return f"Let’s focus on **{topic}** at **{level}** level."
    if topic and rtype:
        return f"I’d like **{topic}** in **{rtype}** format."
    if level and rtype:
        return f"I’m looking for **{rtype}** materials at **{level}** level."
    if topic:
        return f"Let’s focus on **{topic}**."
    if level:
        return f"I’m aiming for **{level}** level."
    if rtype:
        return f"I prefer **{rtype}**."
    return "Here’s what works for me."


def more_same_cluster(context_row, seen_titles, top_k=3):
    """
    Same cluster + same level + new angles
    """
    if not context_row or not seen_titles:
        return pd.DataFrame()
    uh = []
    for t in seen_titles:
        uh.append({
            "title": t,
            "learner_level": context_row["learner_level"],
            "relevant_topic": context_row["relevant_topic"],
            "sent_clusters": context_row["sent_clusters"],
        })
    if uh:
        uh[-1] = {**uh[-1], **{k: v for k, v in context_row.items() if v is not None}}
    return recommend_next_step_from_history(uh, top_k=top_k)


st.title("🎓 Smart Course Recommender Chatbot")
st.caption(
    "A friendly assistant that suggests courses and learning materials matched to your topic, level, and how you like to study."
)
st.divider()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "welcome_inserted" not in st.session_state:
    st.session_state.welcome_inserted = True
    st.session_state.chat_history.insert(
        0,
        {"speaker": "Bot", "message": WELCOME_MESSAGE},
    )

if "user_history" not in st.session_state:
    st.session_state.user_history = []

if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

if "clarify_fields" not in st.session_state:
    st.session_state.clarify_fields = []

if "awaiting_more_offer" not in st.session_state:
    st.session_state.awaiting_more_offer = False

if "recs_pool" not in st.session_state:
    st.session_state.recs_pool = None

if "shown_titles" not in st.session_state:
    st.session_state.shown_titles = set()

if "context_row" not in st.session_state:
    st.session_state.context_row = None

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

user_prompt = st.chat_input("Tell me what you’d like to learn…")

if user_prompt:
    st.session_state.chat_history.append(("User", user_prompt))

    # Follow-up after a recommendation: user declines more (check before "more" — avoids "nothing more" → yes)
    if (
        st.session_state.awaiting_more_offer
        and wants_decline(user_prompt)
        and len(user_prompt) < 160
    ):
        append_bot_message(
            "No problem — I’ll leave it there. Whenever you want to explore something new, just say so."
        )
        st.session_state.awaiting_more_offer = False
        st.session_state.recs_pool = None
        st.session_state.context_row = None
        st.session_state.shown_titles = set()
        st.session_state.last_intent = None

    elif (
        st.session_state.awaiting_more_offer
        and st.session_state.recs_pool is not None
        and st.session_state.context_row is not None
        and wants_more(user_prompt)
        and not looks_like_fresh_topic(user_prompt)
    ):
        pool = st.session_state.recs_pool.reset_index(drop=True)
        shown = set(st.session_state.shown_titles)
        extra = []
        for _, row in pool.iterrows():
            if row["title"] not in shown and len(extra) < 3:
                extra.append(row)
                shown.add(row["title"])

        extra_df = pd.DataFrame(extra) if extra else pd.DataFrame()

        if len(extra) < 2:
            more = more_same_cluster(
                st.session_state.context_row,
                shown,
                top_k=3,
            )
            if not more.empty:
                for _, r in more.iterrows():
                    if r["title"] not in shown and len(extra) < 3:
                        extra.append(r)
                        shown.add(r["title"])
                extra_df = pd.DataFrame(extra) if extra else extra_df

        li = st.session_state.get("last_intent")
        if extra_df is not None and not extra_df.empty:
            if level_stated_by_user(li):
                intro = (
                    "Sure thing — here are a few more at the same level, "
                    "each from a slightly different angle:\n\n"
                )
            else:
                intro = (
                    "Sure thing — here are a few more picks in a similar vein "
                    "(difficulty labels come from the catalog):\n\n"
                )
            msg = (
                intro
                + format_extra_rows(extra_df, intent=li)
                + "\n\nWhen you’re ready for a totally new direction, just type it."
            )
        else:
            msg = (
                "I’m running low on fresh picks in that same spot. "
                "Try rephrasing, or ask about something else — happy to help."
            )

        st.session_state.shown_titles = shown
        if extra_df is not None and not extra_df.empty:
            for _, r in extra_df.iterrows():
                st.session_state.user_history.append(
                    {
                        "title": r["title"],
                        "description": r.get("description", ""),
                        "relevant_topic": r["relevant_topic"],
                        "learner_level": r["learner_level"],
                        "resource_type": r["resource_type"],
                        "sent_clusters": r["sent_clusters"],
                    }
                )

        st.session_state.awaiting_more_offer = False
        st.session_state.recs_pool = None
        st.session_state.context_row = None
        st.session_state.last_intent = None
        append_bot_message(msg)

    else:
        # New recommendation request (or user ignored the follow-up with a full new question)
        intent, recs, needs_clarification = smart_recommend_from_prompt(user_prompt)
        st.session_state.awaiting_more_offer = False
        st.session_state.recs_pool = None
        st.session_state.context_row = None
        st.session_state.shown_titles = set()

        if needs_clarification:
            st.session_state.awaiting_clarification = True
            st.session_state.pending_prompt = user_prompt
            st.session_state.clarify_fields = needs_clarification
            st.session_state.last_intent = intent
            field_help = {
                "topic": "the topic",
                "level": "the difficulty level",
                "resource_type": "the format (book, video, paper, etc.)",
            }
            unsure_parts = [field_help[f] for f in needs_clarification if f in field_help]
            unsure_text = ", ".join(unsure_parts)
            guess_bits = []
            if "topic" in needs_clarification:
                guess_bits.append(f"**{friendly_text(intent['topic'])}**")
            if "level" in needs_clarification:
                guess_bits.append(f"**{friendly_text(intent['level'])}**")
            if "resource_type" in needs_clarification:
                guess_bits.append(f"**{friendly_resource_type(intent['resource_type'])}**")
            if len(guess_bits) == 1:
                guess_line = f"My best guess for that is {guess_bits[0]} — but I’d rather you choose."
            elif len(guess_bits) == 2:
                guess_line = (
                    f"I’m tentatively thinking {guess_bits[0]} and {guess_bits[1]} — "
                    f"pick what fits you below."
                )
            else:
                guess_line = (
                    f"I’m tentatively thinking {guess_bits[0]}, {guess_bits[1]}, and {guess_bits[2]} — "
                    f"adjust below if that’s off."
                )
            append_bot_message(
                f"Hmm — I’m not fully sure about {unsure_text}. {guess_line}"
            )
        elif recs.empty:
            st.session_state.last_intent = None
            append_bot_message(
                "I couldn’t find a great match for that just yet. "
                "Maybe try different words, or another topic — I’ll keep digging."
            )
        else:
            primary = recs.iloc[0]
            st.session_state.last_intent = intent
            st.session_state.recs_pool = recs
            st.session_state.context_row = row_to_context(primary)
            st.session_state.shown_titles = {primary["title"]}
            st.session_state.awaiting_more_offer = True

            st.session_state.user_history.append(
                {
                    "title": primary["title"],
                    "description": primary.get("description", ""),
                    "relevant_topic": primary["relevant_topic"],
                    "learner_level": primary["learner_level"],
                    "resource_type": primary["resource_type"],
                    "sent_clusters": primary["sent_clusters"],
                }
            )

            append_bot_message(
                format_recommendation_body(primary, intent=intent)
                + invite_more_text(intent)
            )

# display conversation
for entry in st.session_state.chat_history:
    if isinstance(entry, dict):
        speaker = entry.get("speaker", "Bot")
        message = entry.get("message", "")
        with st.chat_message("user" if speaker == "User" else "assistant"):
            st.markdown(message)
    else:
        speaker, message = entry
        with st.chat_message("user" if speaker == "User" else "assistant"):
            st.markdown(message)

with st.sidebar:
    st.subheader("Your learning path so far")
    if st.session_state.user_history:
        for idx, item in enumerate(st.session_state.user_history, start=1):
            st.markdown(
                f"{idx}. **{item['title']}**  \n"
                f"{friendly_text(item['relevant_topic'])} · "
                f"{friendly_text(item['learner_level'])} · "
                f"{friendly_resource_type(item['resource_type'])}"
            )
    else:
        st.write("Your picks will show up here as we chat.")

if st.session_state.awaiting_clarification and st.session_state.pending_prompt:
    st.caption("Quick picks — choose what fits you:")
    overrides = {}
    if "topic" in st.session_state.clarify_fields:
        overrides["topic"] = st.selectbox(
            "Topic",
            options=AVAILABLE_TOPICS,
            key="clarify_topic",
        )
    if "level" in st.session_state.clarify_fields:
        overrides["level"] = st.selectbox(
            "Level",
            options=AVAILABLE_LEVELS,
            key="clarify_level",
        )
    if "resource_type" in st.session_state.clarify_fields:
        overrides["resource_type"] = st.selectbox(
            "Resource type",
            options=AVAILABLE_TYPES,
            format_func=friendly_resource_type,
            key="clarify_type",
        )

    if st.button("Apply", key="apply_clarification"):
        st.session_state.chat_history.append(
            ("User", format_preferences_user_message(overrides))
        )
        intent, recs, _ = smart_recommend_from_prompt(
            st.session_state.pending_prompt,
            overrides=overrides,
        )
        st.session_state.awaiting_clarification = False
        st.session_state.pending_prompt = ""
        st.session_state.clarify_fields = []

        if recs.empty:
            st.session_state.last_intent = None
            append_bot_message(
                "Thanks — still nothing solid with those picks. Fancy trying different options below?"
            )
        else:
            primary = recs.iloc[0]
            st.session_state.last_intent = intent
            st.session_state.recs_pool = recs
            st.session_state.context_row = row_to_context(primary)
            st.session_state.shown_titles = {primary["title"]}
            st.session_state.awaiting_more_offer = True
            st.session_state.user_history.append(
                {
                    "title": primary["title"],
                    "description": primary.get("description", ""),
                    "relevant_topic": primary["relevant_topic"],
                    "learner_level": primary["learner_level"],
                    "resource_type": primary["resource_type"],
                    "sent_clusters": primary["sent_clusters"],
                }
            )
            append_bot_message(
                "Nice — here’s what I’d go with:\n\n"
                + format_recommendation_body(primary, intent=intent)
                + invite_more_text(intent)
            )
        st.rerun()
