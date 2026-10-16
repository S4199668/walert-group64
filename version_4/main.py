
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chain
import config
import db
import ingest
import sessions
from retriever import retriever


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


# --------------------------------------------------
# Application startup
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    retriever.reload()
    yield


app = FastAPI(
    title="RMIT Student Support RAG Chatbot",
    lifespan=lifespan,
)


# --------------------------------------------------
# Static files: CSS and other frontend assets
# --------------------------------------------------

if not STATIC_DIR.is_dir():
    raise RuntimeError(
        f"Static directory not found: {STATIC_DIR}"
    )

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# --------------------------------------------------
# Request models
# --------------------------------------------------

class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


class FolderIn(BaseModel):
    path: str | None = None


# --------------------------------------------------
# Frontend
# --------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    html_file = STATIC_DIR / "index.html"

    if not html_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="index.html not found in the static directory",
        )

    return FileResponse(html_file)


# --------------------------------------------------
# Chat
# --------------------------------------------------

@app.post("/api/chat")
def api_chat(body: ChatIn):
    message = body.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Empty message",
        )

    session_id = body.session_id or sessions.new_id()

    return chain.chat(session_id, message)


# --------------------------------------------------
# Document upload and embedding
# --------------------------------------------------

@app.post("/api/upload")
def api_upload(files: list[UploadFile] = File(...)):
    result = {}

    try:
        for file in files:
            filename = file.filename or "unnamed_file"

            if not filename.lower().endswith((".txt", ".pdf")):
                result[filename] = "skipped (only .txt/.pdf)"
                continue

            content = file.file.read()

            result[filename] = ingest.ingest_file(
                filename,
                content,
            )

        retriever.reload()

        return {"chunks_added": result}

    finally:
        for file in files:
            file.file.close()


# --------------------------------------------------
# Ingest documents from a folder
# --------------------------------------------------

@app.post("/api/ingest/folder")
def api_ingest_folder(body: FolderIn):
    folder = (
        Path(body.path)
        if body.path
        else config.DEFAULT_FOLDER
    )

    if not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Folder not found: {folder}",
        )

    result = ingest.ingest_folder(folder)
    retriever.reload()

    return {"chunks_added": result}


# --------------------------------------------------
# Chat session history
# --------------------------------------------------

@app.get("/api/sessions")
def api_sessions():
    output = []

    for session in db.list_sessions():
        history = sessions.load(session["id"])

        first_message = next(
            (
                message["content"]
                for message in history
                if message["role"] == "user"
            ),
            "New chat",
        )

        output.append({
            "id": session["id"],
            "title": first_message[:50],
            "updated": session["updated"],
            "message_count": session["message_count"],
        })

    return output


@app.get("/api/sessions/{sid}")
def api_session(sid: str):
    return sessions.load(sid)


# --------------------------------------------------
# Document/database statistics
# --------------------------------------------------

@app.get("/api/documents")
def api_documents():
    return db.stats()
