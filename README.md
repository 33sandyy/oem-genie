
# OEM Ginie

**OEM Ginie** is a chatbot designed to assist technicians or staff using OEM manuals.  
This project currently includes the **Toyota Malaysia Fortuner OEM Manuals**.

## Project Structure

```
oem_chatbot/
├─ src/
│  ├─ __init__.py
│  ├─ data_ingestion.py
│  ├─ chunker.py
│  ├─ embedder.py
│  ├─ vectorstore_chroma.py
│  ├─ query_engine.py
├─ app.py
├─ build_index.py        # CLI to (re)build DB
├─ requirements.txt
└─ README.md
```

## Setup & Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd oem_chatbot
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Build Database

To ingest PDFs and build the vector database, run:

```bash
python build_index.py --pdf_folder data --persist_dir chroma_db
```

- `--pdf_folder`: Folder containing OEM manual PDFs  
- `--persist_dir`: Directory where the Chroma vector database will be stored

## Run Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will start a local server and you can interact with the OEM Ginie chatbot.

## License

[I will add my license here]