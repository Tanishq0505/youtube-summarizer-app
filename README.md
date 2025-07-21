# YouTube Video Summarizer + AI Chat (LangChain + Gemini)

This Streamlit-based app allows users to:
- Download and transcribe YouTube video subtitles
- Generate intelligent summaries using **Google Gemini**
- Translate summaries into multiple languages
- Export summaries as PDF or TXT
- Email the summary directly to any address
- Engage in Q&A with the video using **LangChain QA chatbot** with memory

> Powered by: `LangChain`, `Google Gemini`, `FAISS`, `Streamlit`, and `yt-dlp`.

---

## Features

- Download subtitles in `.vtt` format from YouTube
- Generate summaries with **Google Gemini**
- Translate to **Hindi, Spanish, French, German**, and more
- Export summaries as `TXT` or `PDF`
- Email the exported summary to yourself or others
- Ask contextual questions using **LangChain QA** with memory & embeddings

---

## Technologies Used

| Technology           | Purpose                                      |
|----------------------|----------------------------------------------|
| Streamlit            | Web app interface                            |
| yt-dlp               | Download captions from YouTube               |
| webvtt               | Parse `.vtt` subtitle files                  |
| Google Gemini        | Summarization & translation via Gemini API   |
| FAISS                | Store document embeddings                    |
| LangChain            | Conversational Retrieval with memory         |
| FPDF                 | PDF export                                   |
| smtplib              | Email delivery with attachment support       |

---

## Project Structure

```plaintext
YT/
├── app.py                      # Main application
├── captions/                   # Stores downloaded captions
├── .streamlit/
│   └── secrets.toml            # API keys and email credentials
├── summary.txt                 # Exported summary (optional)
├── summary.pdf                 # Exported summary (optional)
└── venv/                       # Python virtual environment

## Setup Secrets (`.streamlit/secrets.toml`)

Create a `.streamlit` folder in your project root, and add a `secrets.toml` file with the following:

GEMINI_API_KEY = "Your api key "
EMAIL_USER = "Your mail id"
EMAIL_PASS = "your app password"
For Gmail, use an App Password instead of your regular password.


Installation & Setup
git clone https://github.com/yourusername/youtube-summarizer-app.git
cd youtube-summarizer-app

2. Create Virtual Environment
# Windows
python -m venv venv
venv\Scripts\activate


3. Install Required Packages
pip install -r requirements.txt

4. Run the App
streamlit run app.py

How to Use
Enter a valid YouTube video URL
Type your prompt (e.g., "Summarize the video" or "List key insights")
Optionally select subtitle language and translation language
Enter an email address to send the summary
Click "Generate Summary"
After summary is generated, ask questions and get contextual answers from the transcript

Supported Languages
Captions: English (en), Hindi (hi), Spanish (es), French (fr), German (de), Japanese (ja), Chinese (zh)
Translations: Hindi, Spanish, French, German, or None(English)

Email Support
Email is sent using Gmail SMTP via SSL
Attachments supported: PDF (default)


Author
Tanishq Anand
tanishqanand26@gmail.com
