import streamlit as st, pandas as pd, openai, PyPDF2
from langchain.chat_models import ChatOpenAI

st.image("logo.png", width=320)
st.markdown("Extract verbatim key statements from scientific papers")
st.markdown(":copyright: v0.1 2025 by B. Wohlfarth for IML Institute of Medical Education, University of Bern")

files = st.file_uploader(" ", type="pdf", accept_multiple_files=True)
if st.button("Analysieren") and files:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rows = []
    for f in files:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join(p.extract_text() for p in reader.pages[:50])  # bis 50 Seiten
        prompt = (f"Extract ONE verbatim quote (1–2 sentences) that conveys the central message of this article. "
          f"Also, in another cell specify where in the article (e.g., page number or section) the quote appears and Do NOT include any explanatory text. Only output the quote and the location, on two separate lines.\n\n{text}")
        quote = model.invoke(prompt).content.strip()
        rows.append({"Artikel": f.name, "Zitat": quote})

    df = pd.DataFrame(rows)
    st.subheader("Ergebnis")
    st.dataframe(df, use_container_width=True)
    st.download_button("CSV herunterladen", df.to_csv(index=False), "key_messages.csv")
