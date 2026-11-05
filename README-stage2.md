# Mini ClassBuild: Stage 2 (Steps 6-9)

These files are added next to the Stage 1 project. No Stage 1 file is changed.

## Install
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-stage2.txt
```

## Run
```powershell
python -m streamlit run app_stage2.py
```
`app.py` still runs the Stage 1 version. `app_stage2.py` is the same app plus:
PDF upload and chunking (Step 6), semantic search (Step 7), PDF-grounded lessons with
page citations (Step 8) and a chat under each lesson (Step 9).

The `classbuild/` package contains its own copy of the Stage 1 generation logic, so Stage 2
works without touching `generator.py`, `models.py`, `llm_client.py` or `grading.py`.
It reads `GROQ_API_KEY` and `GROQ_MODEL` from the same `.env` file.

## Tests (offline, no API key needed)
```powershell
python -m unittest discover -s tests -t . -v
```

## Notes
- The first PDF search downloads the embedding model (about 90 MB), so you need internet once.
- No OCR: scanned or image-only PDFs report that no text was found.
- Optional: add `maxUploadSize = 10` under a `[server]` section in `.streamlit/config.toml`.
  The app already rejects PDFs above 10 MB without it.
