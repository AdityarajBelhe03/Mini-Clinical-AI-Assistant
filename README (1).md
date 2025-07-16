# 🩺 Clinical AI Assistant – LangGraph-Inspired Modular Workflow

A lightweight, modular Clinical AI Assistant that processes patient–doctor voice recordings, extracts structured clinical data, and generates a professional SOAP note — powered by Whisper, DeepSeek R1, sentence-transformers, and Pinecone vector DB.

---

## 🚀 Features

- 🎙️ **Audio Transcription** using OpenAI Whisper (or ElevenLabs voiceover)
- 🧠 **Clinical Info Extraction** via prompt-engineered DeepSeek R1
- 📦 **Strict JSON Mapping** of all structured fields
- 🔢 **Vector Embedding + Pinecone DB** storage by medical category
- 🔍 **Semantic Retrieval** of similar cases using RAG
- 📄 **SOAP Note Generation** in markdown, following professional format
- 🌐 **Gradio UI Demo** with visual pipeline walkthrough
- ✅ Includes `confidence_score` for extracted fields

---

## 🖼️ System Architecture

```mermaid
graph TD
    A[Audio Upload] --> B[Transcription (Whisper)]
    B --> C[Clinical Extraction (DeepSeek)]
    C --> D[JSON Structuring]
    D --> E[Embedding (MiniLM-L12-v2)]
    E --> F[Pinecone Vector Storage]
    F --> G[RAG Retrieval]
    G --> H[SOAP Note Generation (DeepSeek)]
```

---

## 🧰 Tech Stack

| Component | Tool |
|----------|------|
| Transcription | Whisper |
| LLM | DeepSeek R1 via OpenRouter |
| Embeddings | sentence-transformers (MiniLM-L12-v2) |
| Vector DB | Pinecone |
| UI | Gradio |
| Language | Python 3.11 |

---

## 📂 Directory Structure

```
clinical_ai_assistant/
├── clinical_outputs/
│   ├── Transcript.txt
│   ├── structured_output.json
│   ├── SOAP_Note_<name>.md
├── SOAP_Agent.py
├── README.md
├── app.py                 # Gradio UI Demo
├── semantic_search.py     # Pinecone Query Logic
```

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python app.py
```

Use the Gradio UI to test the assistant end-to-end.

---

## 🧪 Deliverables

| File | Description |
|------|-------------|
| `Transcript.txt` | Transcribed voiceover using Whisper |
| `structured_output.json` | Structured medical data |
| `SOAP_Note_<name>.md` | Final professional SOAP note |
| `confidence_score` | For each extracted field |
| `Gradio UI` | Demo of the system |
| `semantic_search.py` | Similar case retrieval logic |

---

## 📌 Prompt Engineering Samples

### Clinical Extraction Prompt
> Extract the following categories from the transcript: demographics, chief complaint, diagnosis, tests, medications.

### SOAP Note Prompt
> You are a medical scribe. Use transcript + JSON to write a structured SOAP note. Include clinical reasoning, formal tone, and confidence levels.

---

## 📈 Scaling Plan (Production-Ready)

- Agents modularized as microservices
- Dockerized pipeline orchestrated via LangGraph or Airflow
- Embedding shards by specialty
- PHI-compliant LLM if deployed in healthcare production

---

## 👨‍⚕️ Author

Made with ❤️ by Abhiraj  
📧 abhiraj.gadade@uma.edu.pe

---

## 📄 License

MIT License