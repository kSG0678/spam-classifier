import streamlit as st
import joblib


# -----------------------------
# Load model
# -----------------------------

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="centered"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="title">🛡️ SpamShield AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered Spam Message Detection'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Message input
# -----------------------------

st.subheader("📩 Enter Your Message")

message = st.text_area(
    "Message",
    height=180,
    placeholder="Example: Congratulations! You have won a free prize..."
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Analyze Message", use_container_width=True):

    if not message.strip():

        st.warning("Please enter a message first.")

    else:

        # Convert message to numerical features
        message_vector = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(message_vector)[0]

        # Probability
        probabilities = model.predict_proba(message_vector)[0]

        spam_probability = probabilities[1]
        safe_probability = probabilities[0]


        # -----------------------------
        # Display result
        # -----------------------------

        st.divider()

        st.subheader("🔎 Analysis Result")


        if prediction == 1:

            st.error("🚨 SPAM MESSAGE")

            st.metric(
                "Spam Probability",
                f"{spam_probability * 100:.2f}%"
            )

            st.progress(spam_probability)

            st.write(
                "⚠️ This message contains patterns commonly "
                "associated with spam messages."
            )

        else:

            st.success("✅ NOT SPAM")

            st.metric(
                "Safe Probability",
                f"{safe_probability * 100:.2f}%"
            )

            st.progress(safe_probability)

            st.write(
                "This message appears to be a legitimate message."
            )


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption("SpamShield AI • ML Spam Detection • Version 1.1")

