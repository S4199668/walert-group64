import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import config
import embedder
import sessions
from retriever import retriever

llm = ChatOpenAI(model=config.LLM_MODEL)
verifier_llm = llm.bind(response_format={"type": "json_object"})

CONDENSE_SYS = (
    "Rewrite the user's latest message into a single standalone search query, "
    "using the chat history to resolve pronouns and references. "
    "Do not answer it. Output only the rewritten query."
)

ANSWER_SYS = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
    "If the context does not contain the answer, say you don't have that information. "
    "Be accurate and concise. Do not invent facts."
)

VERIFY_SYS = (
    "You are a strict fact-checker. Given a user query, chat history, retrieved context and a "
    "draft answer, decide whether every factual claim in the answer is supported by the context "
    "(or by the chat history). Respond ONLY with JSON: "
    '{"grounded": true or false, "reason": "short reason"}.'
)

FALLBACK = (
    "I couldn't find reliable information to answer that in my knowledge base. "
    "Could you rephrase or ask something more specific?"
)


def _history_msgs(history):
    recent = history[-config.HISTORY_TURNS * 2:]
    return [HumanMessage(h["content"]) if h["role"] == "user" else AIMessage(h["content"]) for h in recent]


def _history_text(history):
    recent = history[-config.HISTORY_TURNS * 2:]
    return "\n".join(f'{h["role"]}: {h["content"]}' for h in recent) or "(none)"


def _standalone(message, history):
    if not history:  # nothing to resolve, skip call 1
        return message
    out = llm.invoke([SystemMessage(CONDENSE_SYS), *_history_msgs(history), HumanMessage(message)])
    return out.content.strip() or message


def _verify(message, history, context, answer):
    prompt = (
        f"Chat history:\n{_history_text(history)}\n\nUser query: {message}\n\n"
        f"Context:\n{context}\n\nDraft answer:\n{answer}"
    )
    out = verifier_llm.invoke([SystemMessage(VERIFY_SYS), HumanMessage(prompt)])
    try:
        return bool(json.loads(out.content).get("grounded", False))
    except Exception:
        return False


def chat(session_id: str, message: str) -> dict:
    with sessions.lock(session_id):
        history = sessions.load(session_id)

        standalone = _standalone(message, history)             # call 1
        qvec = embedder.embed_query(standalone)                # OpenAI embeddings (in-process)
        chunks = retriever.search(standalone, qvec)            # hybrid + reranker service

        if not chunks:
            answer, grounded = FALLBACK, False
        else:
            context = "\n\n".join(f"[{i + 1}] ({c['source']})\n{c['text']}" for i, c in enumerate(chunks))
            answer = llm.invoke([                              # call 2
                SystemMessage(ANSWER_SYS),
                *_history_msgs(history),
                HumanMessage(f"Context:\n{context}\n\nQuestion: {standalone}"),
            ]).content.strip()
            grounded = _verify(message, history, context, answer)   # call 3
            if not grounded:
                answer = FALLBACK

        history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        sessions.save(session_id, history)

    return {
        "session_id": session_id,
        "answer": answer,
        "grounded": grounded,
        "standalone_query": standalone,
        "sources": sorted({c["source"] for c in chunks}),
    }