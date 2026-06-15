import streamlit as st
from summarizer import summarize
from rating_predictor import predict_rating

# ---------- Page Config ----------
st.set_page_config(
    page_title="Product Review Analyzer",
    page_icon="🛍",
    layout="centered"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main {
    background-color: #f9fafb;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-top: 15px;
}
.rating {
    font-size: 28px;
    color: #ffb703;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("🛍 Product Review Analyzer")
st.write(
    "Enter a long product review to automatically generate a concise summary "
    "and predict the overall customer rating."
)

# ---------- Input ----------
review = st.text_area(
    "✍️ Enter Product Review",
    height=220,
    placeholder="Paste or type a detailed product review here..."
)

# ---------- Action ----------
if st.button("🔍 Analyze Review", use_container_width=True):
    if len(review.strip()) < 30:
        st.warning("⚠️ Please enter a longer review (at least 30 characters).")
    else:
        with st.spinner("🧠 Generating summary..."):
            summary = summarize(review)

        with st.spinner("⭐ Predicting rating..."):
            rating = predict_rating(summary)

        # ---------- Output ----------
        st.markdown("### 📌 Generated Summary")
        st.markdown(f"""
        <div class="card">
            {summary}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("###  Predicted Rating")
        stars = "⭐" * int(rating)
        st.markdown(f"""
        <div class="card">
            <div class="rating">{stars}</div>
            <strong>{rating} / 5</strong>
        </div>
        """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("""
<div class="footer">
    NLP Project | Summarization + Rating Prediction<br>
    Built with Streamlit & Transformers
</div>
""", unsafe_allow_html=True)
