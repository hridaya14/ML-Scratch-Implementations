import streamlit as st
from model import Translator


t = Translator()

# Streamlit UI
st.set_page_config(page_title="English to Spanish Translator", page_icon="🌎", layout="centered")

st.title("🌍 English to Spanish Translator")
st.write("Translate text seamlessly with AI-powered machine translation.")

# Input text area
input_text = st.text_area("Enter text in English:", placeholder="Type here...", height=150)

# Translation button
if st.button("Translate", use_container_width=True):
    if input_text.strip():
        translated_text = t.translate(input_text)
        st.success("Translation completed successfully!")
    else:
        st.warning("Please enter some text to translate.")
        translated_text = ""
else:
    translated_text = ""

# Output box
st.text_area("Translated Text (Spanish):", value=translated_text, height=150, disabled=True)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Transformers")
