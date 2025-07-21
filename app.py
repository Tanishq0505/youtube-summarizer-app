import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import yt_dlp
import os
import re
import time
import webvtt
import google.generativeai as genai
from fpdf import FPDF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# ---------- Utility Functions ----------

def get_video_id(url):
    match = re.search(r"v=([^&]+)", url)
    return match.group(1) if match else None

def download_captions(video_url, lang_code='en', output_dir='captions'):
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': [lang_code],
        'subtitlesformat': 'vtt',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s')
    }
    os.makedirs(output_dir, exist_ok=True)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_id = info.get('id')
    return os.path.join(output_dir, f"{video_id}.{lang_code}.vtt"), video_id

def vtt_to_text(vtt_path):
    captions = [caption.text.strip() for caption in webvtt.read(vtt_path)]
    return " ".join(captions)

def call_gemini_api(prompt):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

def translate_text(text, target_lang):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"Translate the following English text into {target_lang}:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text

def export_to_txt(text, filename='summary.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    return filename

def export_to_pdf(text, filename='summary.pdf'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)
    return filename

def send_email_with_attachment(sender_email, sender_password, receiver_email, subject, body, attachment_path):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    with open(attachment_path, "rb") as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
        msg.attach(part)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

# ---------- Streamlit UI ----------

st.title("🎥 YouTube Video Summarizer + AI Chat")

video_url = st.text_input("Enter YouTube Video URL")
user_prompt = st.text_input("Enter your prompt (e.g., 'Summarize the video')")

lang_code = st.selectbox("Select Subtitle Language", ["en", "hi", "es", "fr", "de", "ja", "zh"], index=0)
target_translate_lang = st.selectbox("Translate Summary To (optional)", ["None", "Hindi", "Spanish", "French", "German"])
email_to = st.text_input("📧 Email the Summary To (optional)")

# ---------- State Initialization ----------
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Generate Summary ----------
if st.button("Generate Summary"):
    if video_url and user_prompt:
        vtt_path, video_id = download_captions(video_url, lang_code)

        if video_id != st.session_state.current_video_id:
            transcript = vtt_to_text(vtt_path)
            st.session_state.transcript = transcript
            st.session_state.current_video_id = video_id

            prompt = f"You are an expert assistant.\nUser Request: {user_prompt}\n\nTranscript:\n{transcript}"
            summary = call_gemini_api(prompt)

            if target_translate_lang != "None":
                summary = translate_text(summary, target_translate_lang)

            st.session_state.summary = summary

            pdf_path = export_to_pdf(summary)
            txt_path = export_to_txt(summary)

            if email_to:
                try:
                    send_email_with_attachment(
                        st.secrets["EMAIL_USER"],
                        st.secrets["EMAIL_PASS"],
                        email_to,
                        "Your YouTube Video Summary",
                        "Please find attached the summary.",
                        pdf_path
                    )
                    st.success("✅ Email sent successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to send email: {e}")

            # Recreate retriever and QA chain with new content
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents([Document(page_content=transcript)])
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GEMINI_API_KEY"])
            vectorstore = FAISS.from_documents(docs, embeddings)

            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=st.secrets["GEMINI_API_KEY"])
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)
            st.session_state.chat_history = []

# ---------- Show Summary ----------
if st.session_state.summary:
    st.subheader("📝 Summary:")
    st.write(st.session_state.summary)

    pdf_path = export_to_pdf(st.session_state.summary)
    txt_path = export_to_txt(st.session_state.summary)

    with st.expander("📥 Download Summary"):
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download as TXT", open(txt_path, "rb"), file_name="summary.txt")
        with col2:
            st.download_button("Download as PDF", open(pdf_path, "rb"), file_name="summary.pdf")

# ---------- Ask Questions ----------
st.header("💬 Ask Questions About the Video")

if st.session_state.qa_chain:
    question = st.text_input("Ask your question:")
    if st.button("Ask") and question:
        answer = st.session_state.qa_chain.run(question)
        st.session_state.chat_history.append(("You", question))
        st.session_state.chat_history.append(("Bot", answer))

    for sender, message in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {message}")
