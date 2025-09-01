import os
import json
import base64
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.warning("Setează OPENAI_API_KEY în .env sau ca variabilă de mediu.")

client = OpenAI(api_key=OPENAI_API_KEY)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "book_summaries.json"

def load_books():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

BOOKS = load_books()

def get_summary_by_title(title: str) -> str:
    t = title.strip().lower()
    for b in BOOKS:
        if b["title"].strip().lower() == t:
            return b["summary"]
    return "Nu am găsit rezumat pentru acest titlu."

def is_allowed_by_moderation(text: str) -> bool:
    try:
        resp = client.moderations.create(
            model="text-moderation-latest",
            input=text or ""
        )
        return not resp.results[0].flagged
    except Exception:
        return True

def synthesize_tts(text: str) -> bytes:
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_path = tmp.name
    tmp.close()
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    ) as response:
        response.stream_to_file(tmp_path)
    data = Path(tmp_path).read_bytes()
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return data

def transcribe_bytes_to_text(raw_bytes: bytes) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return (tr.text or "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def generate_cover(title: str, summary: str) -> bytes:
    prompt = (
        f"Minimalist, modern book cover illustration without any text. "
        f"Concept inspired by the book titled '{title}'. "
        f"Use a strong central motif reflecting these ideas: {summary[:400]}. "
        f"Clean composition, high contrast, single focal element, no typography."
    )
    img = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        n=1
    )
    b64 = img.data[0].b64_json
    return base64.b64decode(b64)

import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parent))
from rag_store import ensure_indexed, semantic_search

st.set_page_config(page_title="Smart Librariann using RAG and ChromaDB", page_icon="📚", layout="centered")
st.title("Smart Librariann using RAG and ChromaDB")

with st.spinner("Verific indexarea în ChromaDB…"):
    ensure_indexed()

with st.expander("Vezi Arhiva", expanded=False):
    st.json(BOOKS)

if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "rec" not in st.session_state:
    st.session_state.rec = None
if "title_match" not in st.session_state:
    st.session_state.title_match = None
if "full_text" not in st.session_state:
    st.session_state.full_text = None
if "cover_cache" not in st.session_state:
    st.session_state.cover_cache = {}

mic_audio = mic_recorder(
    start_prompt="🎤 Înregistrează",
    stop_prompt="■ Oprește",
    just_once=True,
    key="mic_rec"
)

if mic_audio and isinstance(mic_audio, dict) and mic_audio.get("bytes"):
    try:
        transcript = transcribe_bytes_to_text(mic_audio["bytes"])
        if transcript:
            st.session_state.query_text = transcript
            st.success("Transcriere realizată.")
        else:
            st.error("Transcriere goală.")
    except Exception as e:
        st.error(f"Eroare la transcriere: {e}")

query = st.text_input("Ce fel de carte cauți?", value=st.session_state.query_text)
k = st.slider("Top-k în căutarea semantică", 1, 5, 3)

def llm_recommend(user_query: str, hits: list[dict]) -> str:
    system_prompt = (
        "Ești un asistent bibliotecar. Pe baza fragmentelor recuperate, recomandă O SINGURĂ carte potrivită. "
        "Răspunde concis în română: titlul recomandat pe prima linie, apoi 2–4 motive pe linii noi. "
        "Nu inventa titluri și nu recomanda mai multe deodată."
    )
    context = "\n\n".join([f"- {h['title']}: {h['doc']}" for h in hits])
    user_prompt = f"Întrebare: {user_query}\n\nCărți candidate:\n{context}\n\nAlege o singură carte din cele de mai sus."

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

if st.button(" Caută și recomandă") and (query or st.session_state.query_text):
    effective_query = query or st.session_state.query_text
    if not is_allowed_by_moderation(effective_query):
        st.warning("Mesajul tău a fost blocat de filtrul de conținut. Te rog reformulează.")
    else:
        with st.spinner("Caut în ChromaDB și generez recomandarea…"):
            hits = semantic_search(effective_query, k=k)
            if not hits:
                st.error("Nu am găsit potriviri în colecție.")
            else:
                rec = llm_recommend(effective_query, hits)
                titles = [h["title"] for h in hits]
                title_match = None
                for t in titles:
                    if t.lower() in rec.lower():
                        title_match = t
                        break
                if not title_match:
                    title_match = rec.splitlines()[0].strip().strip("„”\"' ")
                full_text = get_summary_by_title(title_match)
                st.session_state.rec = rec
                st.session_state.title_match = title_match
                st.session_state.full_text = full_text
                st.session_state.query_text = effective_query

if st.session_state.rec and st.session_state.title_match and st.session_state.full_text:
    st.subheader(" Recomandare")
    st.write(st.session_state.rec)
    st.markdown("---")
    st.subheader(f"Rezumat detaliat: {st.session_state.title_match}")
    st.write(st.session_state.full_text)

    if st.button("Ascultă recomandarea și rezumatul", key="tts_button"):
        try:
            audio_bytes = synthesize_tts(f"Recomandare: {st.session_state.rec}\n\nRezumat: {st.session_state.full_text}")
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.error(f"Nu am putut genera audio: {e}")

    if st.button("Vezi coperta", key="cover_btn"):
        try:
            title_key = st.session_state.title_match.strip().lower()
            if title_key in st.session_state.cover_cache:
                img_bytes = st.session_state.cover_cache[title_key]
            else:
                img_bytes = generate_cover(st.session_state.title_match, st.session_state.full_text)
                st.session_state.cover_cache[title_key] = img_bytes
            st.image(img_bytes, caption=f"Copertă generată pentru „{st.session_state.title_match}”")
        except Exception as e:
            st.error(f"Nu am putut genera imaginea: {e}")

st.markdown("---")
