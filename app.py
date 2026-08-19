"""
AI Knowledge Assistant
-----------------------
A single-file Streamlit application that recreates the AI Knowledge Assistant:
    - Welcome / Home
    - Dashboard (live stats & charts)
    - Chatbot (ask questions, optionally powered by Google Gemini)
    - Resources (curated learning links by category)
    - Categories (browse knowledge categories)
    - History (full searchable chat history)

Run with:
    streamlit run app.py

Optional: set an environment variable GEMINI_API_KEY to enable real AI answers
via Google's Gemini API (requires `pip install google-generativeai`). Without
a key, the chatbot falls back to a lightweight built-in knowledge base.
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_assistant.db")

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# STATIC KNOWLEDGE BASE (categories + resources)
# --------------------------------------------------------------------------
CATEGORIES = {
    "Science": {
        "icon": "🔬",
        "description": "Physics, chemistry, biology & natural sciences",
        "resources": [
            ("Khan Academy Science", "https://www.khanacademy.org/science", "Free science courses for every level"),
            ("NASA Science", "https://science.nasa.gov/", "Space, earth & planetary science"),
            ("PhET Interactive Simulations", "https://phet.colorado.edu/", "Hands-on science simulations"),
        ],
    },
    "Technology": {
        "icon": "💻",
        "description": "Software, hardware, networking & modern tech",
        "resources": [
            ("MDN Web Docs", "https://developer.mozilla.org/", "The definitive web technology reference"),
            ("freeCodeCamp", "https://www.freecodecamp.org/", "Free coding & tech curriculum"),
            ("Cisco Networking Academy", "https://www.netacad.com/", "Learn networking fundamentals"),
        ],
    },
    "Programming": {
        "icon": "🐍",
        "description": "Learn to build software across languages",
        "resources": [
            ("Official Python Docs", "https://docs.python.org/3/", "Complete Python language reference"),
            ("Real Python", "https://realpython.com/", "Practical Python tutorials for all levels"),
            ("W3Schools Python", "https://www.w3schools.com/python/", "Beginner-friendly Python guide"),
        ],
    },
    "Artificial Intelligence": {
        "icon": "🤖",
        "description": "AI, machine learning, deep learning & neural networks",
        "resources": [
            ("Google Machine Learning Crash Course", "https://developers.google.com/machine-learning/crash-course", "Fast-paced intro to ML"),
            ("fast.ai", "https://www.fast.ai/", "Practical deep learning for coders"),
            ("Hugging Face Course", "https://huggingface.co/learn", "Learn modern NLP & transformers"),
        ],
    },
}

# Simple built-in fallback knowledge base for the chatbot (used when no
# Gemini API key is configured).
FALLBACK_ANSWERS = {
    "python": "Python is a high-level, general-purpose programming language known for its readable syntax.",
    "what is python": "Python is a high-level, general-purpose programming language known for its readable syntax.",
    "ml": "Machine Learning (ML) is a branch of AI where systems learn patterns from data instead of being explicitly programmed.",
    "what is ml": "Machine Learning (ML) is a branch of AI where systems learn patterns from data instead of being explicitly programmed.",
    "ai": "Artificial Intelligence (AI) is the field of building systems that can perform tasks normally requiring human intelligence.",
    "hello": "Hello! I'm your AI Knowledge Assistant. Ask me about Python, ML, AI, or explore the Resources and Categories pages.",
}


# --------------------------------------------------------------------------
# DATABASE HELPERS
# --------------------------------------------------------------------------
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            category TEXT DEFAULT 'General',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def log_chat(question: str, answer: str, category: str = "General"):
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_history (question, answer, category, created_at) VALUES (?, ?, ?, ?)",
        (question, answer, category, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def fetch_history(search: str = "") -> pd.DataFrame:
    conn = get_connection()
    if search:
        df = pd.read_sql_query(
            "SELECT * FROM chat_history WHERE question LIKE ? OR answer LIKE ? ORDER BY id DESC",
            conn,
            params=(f"%{search}%", f"%{search}%"),
        )
    else:
        df = pd.read_sql_query("SELECT * FROM chat_history ORDER BY id DESC", conn)
    conn.close()
    return df


def get_metrics():
    df = fetch_history()
    total_questions = len(df)
    total_categories = len(CATEGORIES)
    total_resources = sum(len(c["resources"]) for c in CATEGORIES.values())
    db_online = True
    return total_questions, total_categories, total_resources, db_online, df


# --------------------------------------------------------------------------
# CHATBOT LOGIC
# --------------------------------------------------------------------------
def guess_category(question: str) -> str:
    q = question.lower()
    if "python" in q or "code" in q or "program" in q:
        return "Programming"
    if "ml" in q or "machine learning" in q or "model" in q:
        return "Artificial Intelligence"
    if "ai" in q or "neural" in q or "deep learning" in q:
        return "Artificial Intelligence"
    if "science" in q or "physics" in q or "biology" in q:
        return "Science"
    if "network" in q or "software" in q or "hardware" in q:
        return "Technology"
    return "General"


def ask_gemini(question: str) -> str:
    """Try to answer using Google Gemini if an API key is configured."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(question)
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001
        return f"(Gemini error, falling back to local answers: {exc})"


def get_answer(question: str) -> str:
    gemini_answer = ask_gemini(question)
    if gemini_answer and not gemini_answer.startswith("(Gemini error"):
        return gemini_answer

    key = question.strip().lower()
    if key in FALLBACK_ANSWERS:
        return FALLBACK_ANSWERS[key]
    for k, v in FALLBACK_ANSWERS.items():
        if k in key:
            return v
    return (
        "I don't have a specific answer for that yet, but you can explore the "
        "Resources and Categories pages to dig deeper, or set a GEMINI_API_KEY "
        "environment variable to enable full AI-powered answers."
    )


# --------------------------------------------------------------------------
# PAGES
# --------------------------------------------------------------------------
def page_home():
    st.title("🧠 Welcome to AI Knowledge Assistant")
    st.markdown("Use the sidebar to navigate between pages.")

    st.table(
        pd.DataFrame(
            {
                "Page": ["📊 Dashboard", "💬 Chatbot", "📚 Resources", "📂 Categories", "📝 History"],
                "Description": [
                    "Live stats, charts & recent conversations",
                    "Ask the AI anything",
                    "Curated learning links by topic",
                    "Browse knowledge categories",
                    "Full searchable chat history",
                ],
            }
        )
    )
    st.info("👉 Select a page from the sidebar to get started.")


def page_dashboard():
    st.title("📊 AI Knowledge Assistant — Dashboard")
    st.caption(f"Last refreshed: {datetime.now().strftime('%B %d, %Y %H:%M')}")
    st.divider()

    total_questions, total_categories, total_resources, db_online, df = get_metrics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💬 Total Questions", total_questions)
    c2.metric("📂 Categories", total_categories)
    c3.metric("📚 Resources", total_resources)
    c4.metric("🟢 DB", "Online" if db_online else "Offline")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Questions by Category")
        if not df.empty:
            counts = df["category"].value_counts().reset_index()
            counts.columns = ["Category", "Questions"]
            fig = px.bar(counts, x="Category", y="Questions", color="Category")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No questions asked yet — try the Chatbot page!")

    with col2:
        st.subheader("🥧 Category Share")
        if not df.empty:
            counts = df["category"].value_counts().reset_index()
            counts.columns = ["Category", "Questions"]
            fig = px.pie(counts, names="Category", values="Questions")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No data to show yet.")

    st.divider()
    st.subheader("🕘 Recent Conversations")
    if not df.empty:
        st.dataframe(df.head(5)[["created_at", "question", "answer", "category"]], use_container_width=True)
    else:
        st.write("Nothing here yet.")


def page_chatbot():
    st.title("💬 Chatbot")
    st.caption("Ask the AI anything, powered by Gemini (falls back to a local knowledge base).")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        answer = get_answer(question)
        category = guess_category(question)
        log_chat(question, answer, category)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


def page_resources():
    st.title("📚 Learning Resources")
    st.caption("Curated tutorials and references to level up your skills")
    st.divider()

    search = st.text_input("🔍 Search resources…", placeholder="e.g. Python, deep learning")

    for cat_name, cat in CATEGORIES.items():
        resources = cat["resources"]
        if search:
            resources = [r for r in resources if search.lower() in r[0].lower() or search.lower() in cat_name.lower()]
            if not resources:
                continue

        st.subheader(f"{cat['icon']} {cat_name}")
        cols = st.columns(len(resources)) if resources else []
        for col, (name, url, desc) in zip(cols, resources):
            with col:
                st.markdown(f"**[{name}]({url})**")
                st.caption(desc)
        st.write("")


def page_categories():
    st.title("📂 Knowledge Categories")
    st.caption("Browse topics available in the AI Knowledge Assistant")
    st.divider()

    cat_items = list(CATEGORIES.items())
    for i in range(0, len(cat_items), 2):
        row = cat_items[i : i + 2]
        cols = st.columns(2)
        for col, (cat_name, cat) in zip(cols, row):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {cat['icon']} {cat_name}")
                    st.write(cat["description"])
                    st.caption(f"📌 {len(cat['resources'])} resources")


def page_history():
    st.title("📝 Chat History")
    st.caption("All your previous AI conversations in one place")
    st.divider()

    df = fetch_history()
    st.info(f"📖 {len(df)} conversation(s) found")

    search = st.text_input("🔍 Search history…", placeholder="Search questions or answers")
    df = fetch_history(search)

    if df.empty:
        st.write("No conversations match your search yet.")
        return

    for _, row in df.iterrows():
        with st.expander(f"❓ {row['question']}"):
            st.markdown(f"**Answer:** {row['answer']}")
            st.caption(f"Category: {row['category']}  •  {row['created_at']}")


# --------------------------------------------------------------------------
# MAIN / NAVIGATION
# --------------------------------------------------------------------------
def main():
    st.sidebar.markdown("## 🧠 AI Knowledge Assistant")
    st.sidebar.divider()

    pages = {
        "🏠 Home": page_home,
        "📊 Dashboard": page_dashboard,
        "💬 Chatbot": page_chatbot,
        "📚 Resources": page_resources,
        "📂 Categories": page_categories,
        "📝 History": page_history,
    }

    choice = st.sidebar.radio("Navigation", list(pages.keys()))
    st.sidebar.divider()
    st.sidebar.caption("Built with Streamlit • single-file app.py")

    pages[choice]()


if __name__ == "__main__":
    get_connection().close()  # ensure DB/tables exist on startup
    main()