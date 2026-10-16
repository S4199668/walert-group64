import io
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config
import db
import embedder

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
)


def read_bytes(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


def ingest_text(source: str, text: str) -> int:
    chunks = [c for c in _splitter.split_text(text) if c.strip()]
    if not chunks:
        return 0
    embs = embedder.embed_documents(chunks)
    db.replace_source(source, chunks, embs)
    return len(chunks)


def ingest_file(filename: str, data: bytes) -> int:
    return ingest_text(Path(filename).stem, read_bytes(filename, data))


def ingest_folder(folder) -> dict:
    result = {}
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() in (".txt", ".pdf"):
            result[p.name] = ingest_file(p.name, p.read_bytes())
            print(f"{p.name}: {result[p.name]} chunks")
    return result


if __name__ == "__main__":
    db.init()
    ingest_folder(sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_FOLDER)
    print("Done.")
