# Streamlit YouTube Summarizer + AI Chat 
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import yt_dlp
import os
import re
import unicodedata
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
from langchain_community.vectorstores import FAISS  # ✅ updated import
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# ---------- Utility Functions ----------

def get_video_id(url: str):
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
    # YouTube sometimes names auto-captions differently; try both manual+auto patterns
    manual_path = os.path.join(output_dir, f"{video_id}.{lang_code}.vtt")
    auto_path = os.path.join(output_dir, f"{video_id}.vtt")
    vtt_path = manual_path if os.path.exists(manual_path) else auto_path
    return vtt_path, video_id


def vtt_to_text(vtt_path):
    captions = []
    for caption in webvtt.read(vtt_path):
        # Remove newlines inside caption blocks, strip timing artifacts
        text = caption.text.replace('\n', ' ').strip()
        captions.append(text)
    return ' '.join(captions)


def call_gemini_api(prompt: str) -> str:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return getattr(response, 'text', '').strip()


def translate_text(text: str, target_lang: str) -> str:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"Translate the following English text into {target_lang}:\n\n{text}"
    response = model.generate_content(prompt)
    return getattr(response, 'text', '').strip()


def export_to_txt(text, filename='summary.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    return filename


# ---- PDF helpers (Unicode fonts) ----
FONT_CANDIDATES_BY_LANG = {
    'Hindi': [
        'assets/NotoSansDevanagari-Regular.ttf',
        'NotoSansDevanagari-Regular.ttf',
    ],
    'Japanese': [
        'assets/NotoSansJP-Regular.otf',
        'NotoSansJP-Regular.ttf',
    ],
    'Chinese': [
        'assets/NotoSansSC-Regular.otf',
        'assets/NotoSansSC-Regular.ttf',
        'NotoSansSC-Regular.otf',
        'NotoSansSC-Regular.ttf',
    ],
    'Korean': [
        'assets/NotoSansKR-Regular.otf',
        'NotoSansKR-Regular.otf',
    ],
    # default fallback (broad Latin coverage)
    'Default': [
        'assets/DejaVuSans.ttf',
        'DejaVuSans.ttf',
        'assets/NotoSans-Regular.ttf',
        'NotoSans-Regular.ttf',
    ],
}


def _pick_font_paths(lang_hint: str):
    if lang_hint in FONT_CANDIDATES_BY_LANG:
        return FONT_CANDIDATES_BY_LANG[lang_hint] + FONT_CANDIDATES_BY_LANG['Default']
    return FONT_CANDIDATES_BY_LANG['Default']


def _first_existing_font(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _ascii_fallback(text: str) -> str:
    # replace common dashes and quotes before ascii fold to keep readability
    replacements = {
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\xa0': ' ',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # ASCII fold (drops unsupported chars)
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def export_to_pdf(text: str, filename='summary.pdf', lang_hint: str = 'Default'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Try to add a proper Unicode font
    font_paths = _pick_font_paths(lang_hint)
    chosen = _first_existing_font(font_paths)

    use_ascii_fallback = False
    if chosen:
        try:
            pdf.add_font('UNI', '', chosen, uni=True)
            pdf.set_font('UNI', size=12)
        except Exception:
            # if font load fails for some reason, fallback to core font + ASCII cleaning
            pdf.set_font('Helvetica', size=12)
            use_ascii_fallback = True
    else:
        # Font not found, warn & fallback
        pdf.set_font('Helvetica', size=12)
        use_ascii_fallback = True

    # Define max cell width (page width - 2 * margin)
    page_width = pdf.w - 2 * 15

    # Clean/normalize lines and break very long tokens
    cleaned_lines = []
    if use_ascii_fallback:
        text = _ascii_fallback(text)

    for line in text.split('\n'):
        cleaned_line = []
        for word in line.split(' '):
            if len(word) > 80:
                chunks = [word[i:i+80] for i in range(0, len(word), 80)]
                cleaned_line.extend(chunks)
            else:
                cleaned_line.append(word)
        cleaned_lines.append(' '.join(cleaned_line).strip())

    for line in cleaned_lines:
        pdf.multi_cell(page_width, 10, line)

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
translate_to = st.selectbox("Translate Summary To (optional)", ["None", "Hindi", "Spanish", "French", "German", "Japanese", "Chinese", "Korean"])  # extended
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

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# Map translation target to font hint for PDF
lang_hint_map = {
    'Hindi': 'Hindi',
    'Japanese': 'Japanese',
    'Chinese': 'Chinese',
    'Korean': 'Korean',
}

# ---------- Generate Summary ----------
if st.button("Generate Summary"):
    if not video_url or not user_prompt:
        st.error("Please enter both a YouTube URL and a prompt.")
    else:
        try:
            vtt_path, video_id = download_captions(video_url, lang_code)
            if not os.path.exists(vtt_path):
                st.error("Could not find downloaded captions. Try a different language or video with captions.")
            else:
                transcript = vtt_to_text(vtt_path)
                st.session_state.transcript = transcript
                st.session_state.current_video_id = video_id

                prompt = (
                    "You are an expert assistant.\n"
                    f"User Request: {user_prompt}\n\n"
                    f"Transcript (may contain errors, summarize or act accordingly):\n{transcript}"
                )
                summary = call_gemini_api(prompt)

                if translate_to != "None":
                    summary = translate_text(summary, translate_to)

                st.session_state.summary = summary

                # Pick font hint based on translation target
                lang_hint = lang_hint_map.get(translate_to, 'Default')

                pdf_path = export_to_pdf(summary, filename='summary.pdf', lang_hint=lang_hint)
                txt_path = export_to_txt(summary, filename='summary.txt')

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

        except Exception as e:
            st.error(f"Failed to process video: {e}")

# ---------- Show Summary ----------
if st.session_state.summary:
    st.subheader("📝 Summary:")
    st.write(st.session_state.summary)

    # Re-export to ensure files exist for download
    lang_hint = lang_hint_map.get(translate_to, 'Default')
    pdf_path = export_to_pdf(st.session_state.summary, filename='summary.pdf', lang_hint=lang_hint)
    txt_path = export_to_txt(st.session_state.summary, filename='summary.txt')

    with st.expander("📥 Download Summary"):
        col1, col2 = st.columns(2)
        with col1:
            with open(txt_path, "rb") as f:
                st.download_button("Download as TXT", data=f, file_name="summary.txt")
        with col2:
            with open(pdf_path, "rb") as f:
                st.download_button("Download as PDF", data=f, file_name="summary.pdf")

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
