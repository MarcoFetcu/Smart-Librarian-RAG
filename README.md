# Smart Librarian

## Descriere
Smart Librarian este o aplicație construită cu **Streamlit**, **ChromaDB** și un pipeline de tip **RAG (Retrieval-Augmented Generation)**.  
Aceasta permite recomandarea de cărți în funcție de interesele utilizatorului, utilizând embeddings pentru căutare semantică și un model de limbaj pentru a genera recomandarea finală.

## Funcționalități
- **Căutare carte** prin:
  - introducere text
  - căutare vocală (înregistrare voce cu microfon)
- **Filtru de moderare** pentru limbaj nepotrivit, folosind OpenAI Moderation.
- **Recomandare unică de carte**, cu titlul și argumentele aferente.
- **Rezumat detaliat** al cărții recomandate.
- **Text-to-Speech (TTS)**: redarea audio a recomandării și rezumatului.
- **Generare copertă**: imagine reprezentativă pentru cartea recomandată, generată automat.

## Tehnologii folosite
- [Streamlit](https://streamlit.io/) pentru interfața web
- [ChromaDB](https://www.trychroma.com/) ca vector store
- RAG (Retrieval-Augmented Generation) cu embeddings și OpenAI GPT
- OpenAI API pentru:
  - embeddings
  - filtrare limbaj nepotrivit
  - speech-to-text (Whisper)
  - text-to-speech
  - generare imagini

## Instalare
1. Clonează repository-ul sau descarcă sursele.
2. Creează și activează un mediu virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux / Mac
   .venv\Scripts\activate       # Windows
   ```
3. Instalează dependențele:
   ```bash
   pip install -r requirements.txt
   ```
4. Configurează cheia OpenAI:
   - Creează un fișier `.env` în rădăcina proiectului
   - Adaugă:
     ```
     OPENAI_API_KEY=cheia_ta_aici
     ```

## Rulare
Pornește aplicația cu:
```bash
streamlit run app/streamlit_app.py
```

După pornire, aplicația va fi disponibilă la:
```
http://localhost:8501
```

## Utilizare
- Introdu manual o întrebare (ex: *"Vreau o carte despre libertate și control social"*) sau folosește butonul de microfon pentru a dicta întrebarea.
- Apasă **Caută și recomandă** pentru a primi:
  - recomandarea unei singure cărți relevante
  - rezumatul complet
- Opțional:
  - redă audio recomandarea și rezumatul prin TTS
  - generează o copertă sugestivă pentru cartea recomandată
