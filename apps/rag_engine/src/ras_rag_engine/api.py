"""OpenAI-compatible chat completions API backed by the RAG engine."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .agent import run_agent_streaming
from .citations import extract_content, renumber_response, strip_llm_sources
from .config import RAGConfig
from .retriever import RetrievedChunk

log = logging.getLogger(__name__)

# Load .env for local development (Lambda gets credentials from IAM role)
if not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    from dotenv import load_dotenv

    load_dotenv()

app = FastAPI(title="SwetBot RAG API")
_config: RAGConfig | None = None
_security = HTTPBearer(auto_error=False)


def _get_config() -> RAGConfig:
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def _check_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
    config: RAGConfig = Depends(_get_config),
) -> None:
    if not config.api_key:
        return
    if credentials is None or credentials.credentials != config.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "swetbot"
    messages: list[ChatMessage]
    stream: bool = True
    temperature: float | None = None


@app.get("/")
def healthcheck():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models(config: RAGConfig = Depends(_get_config)):
    return {
        "object": "list",
        "data": [
            {
                "id": "swetbot",
                "object": "model",
                "created": 0,
                "owned_by": "raskl-rag",
            }
        ],
    }


def _build_chunk_event(
    completion_id: str,
    content: str = "",
    finish_reason: str | None = None,
) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "swetbot",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return json.dumps(chunk)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _auth: None = Depends(_check_auth),
    config: RAGConfig = Depends(_get_config),
):
    messages = [m.model_dump() for m in request.messages]
    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    user_message = messages[-1]["content"]
    history = messages[:-1]

    if request.temperature is not None:
        config = config.model_copy(update={"llm_temperature": request.temperature})

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Lambda buffered mode can't relay SSE — Open WebUI sees content-type text/event-stream
    # but receives the full body at once, then forwards raw SSE text to the browser.
    # Force non-streaming on Lambda so Open WebUI gets application/json and handles it correctly.
    on_lambda = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

    if not request.stream or on_lambda:
        return _non_streaming_response(user_message, history, config, completion_id)

    return EventSourceResponse(
        _stream_response(user_message, history, config, completion_id),
        media_type="text/event-stream",
    )


def _non_streaming_response(
    user_message: str,
    history: list[dict],
    config: RAGConfig,
    completion_id: str,
) -> dict:
    all_chunks: list[RetrievedChunk] = []
    final_text = ""
    for partial_text, chunks in run_agent_streaming(user_message, history, config):
        final_text = partial_text
        all_chunks = chunks

    final_text = renumber_response(final_text, all_chunks)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "swetbot",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _stream_response(
    user_message: str,
    history: list[dict],
    config: RAGConfig,
    completion_id: str,
):
    """Yield SSE events in OpenAI streaming format."""
    # Initial role event
    role_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "swetbot",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield {"data": json.dumps(role_chunk)}

    all_chunks: list[RetrievedChunk] = []
    prev_text = ""

    try:
        for partial_text, chunks in run_agent_streaming(user_message, history, config):
            all_chunks = chunks
            # Send incremental content delta
            if len(partial_text) > len(prev_text):
                delta = partial_text[len(prev_text) :]
                prev_text = partial_text
                yield {"data": _build_chunk_event(completion_id, content=delta)}

        # After streaming completes, check if the LLM generated its own sources section.
        # If so, we can't un-send it — skip appending code-generated sources.
        content = extract_content(prev_text)
        clean = strip_llm_sources(content)
        if len(clean) < len(content):
            # LLM generated sources already streamed — skip code sources
            log.info("LLM generated its own sources section; skipping code-generated sources")
        else:
            final_with_citations = renumber_response(prev_text, all_chunks)
            if len(final_with_citations) > len(prev_text):
                citation_delta = final_with_citations[len(prev_text) :]
                yield {"data": _build_chunk_event(completion_id, content=citation_delta)}

    except Exception as e:
        log.exception("Error during streaming")
        yield {"data": _build_chunk_event(completion_id, content=f"\n\nError: {e}")}

    yield {"data": _build_chunk_event(completion_id, finish_reason="stop")}
    yield {"data": "[DONE]"}


# --- Image serving endpoint ---


@app.get("/v1/images/{figure_id}")
async def get_image(
    figure_id: str,
    thumb: bool = Query(False),
    config: RAGConfig = Depends(_get_config),
):
    """Serve a figure image. Local mode returns file, Lambda mode redirects to S3 presigned URL."""
    import psycopg
    from pgvector.psycopg import register_vector

    with psycopg.connect(config.dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.doc_id, f.asset_path, f.thumb_path, d.s3_prefix
                FROM figures f
                JOIN documents d ON f.doc_id = d.doc_id
                WHERE f.figure_id = %s
                """,
                (figure_id,),
            )
            row = cur.fetchone()

    if not row:
        log.warning("Image 404: figure_id=%s not found in database", figure_id)
        raise HTTPException(status_code=404, detail="Figure not found")

    doc_id, asset_path, thumb_path, s3_prefix = row
    rel_path = thumb_path if (thumb and thumb_path) else asset_path

    on_lambda = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

    if on_lambda and config.s3_bucket:
        import boto3

        s3_key = f"{s3_prefix}/{rel_path}" if s3_prefix else f"processed/{doc_id}/latest/{rel_path}"
        s3_client = boto3.client("s3", region_name=config.bedrock_region)
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": config.s3_bucket, "Key": s3_key},
            ExpiresIn=3600,
        )
        return RedirectResponse(url=presigned_url, status_code=302)

    # Local mode: serve from disk
    local_path = Path(config.data_dir) / doc_id / rel_path
    if not local_path.is_file():
        log.warning("Image 404: figure_id=%s, doc_id=%s, path=%s", figure_id, doc_id, local_path)
        raise HTTPException(status_code=404, detail=f"Image file not found: {local_path}")

    from fastapi.responses import FileResponse

    return FileResponse(str(local_path), media_type="image/jpeg")


# --- Audio endpoints (AWS Transcribe + Polly) ---

# OpenAI voice → Polly VoiceId mapping
_POLLY_VOICE_MAP = {
    "alloy": "Arthur",  # British male (neural)
    "echo": "Brian",  # British male (neural)
    "nova": "Amy",  # British female (neural)
    "fable": "Zhiyu",  # Mandarin Chinese female (standard)
    "shimmer": "Hiujin",  # Cantonese female (neural)
    "onyx": "Kajal",  # Indian English/Hindi female (neural)
}

# Voices only available with "standard" engine (not "neural")
_POLLY_STANDARD_ONLY = {"Zhiyu"}


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile,
    model: str = Form("whisper-1"),
    _auth: None = Depends(_check_auth),
    config: RAGConfig = Depends(_get_config),
):
    """OpenAI-compatible STT endpoint backed by AWS Transcribe."""
    if config.llm_provider == "model_studio":
        raise HTTPException(status_code=501, detail="Speech-to-text not yet implemented for Model Studio provider")

    import boto3

    bucket = config.transcribe_s3_bucket
    if not bucket:
        raise HTTPException(status_code=500, detail="CHAT_TRANSCRIBE_S3_BUCKET not configured")

    region = config.bedrock_region
    job_id = f"stt-{uuid.uuid4().hex[:12]}"
    ext = (file.filename or "audio.webm").rsplit(".", 1)[-1] or "webm"
    s3_key = f"tmp/transcribe/{job_id}.{ext}"

    s3 = boto3.client("s3", region_name=region)
    transcribe = boto3.client("transcribe", region_name=region)

    try:
        # Upload audio to S3
        audio_bytes = await file.read()
        s3.put_object(Bucket=bucket, Key=s3_key, Body=audio_bytes)

        # Map file extensions to Transcribe media formats
        media_format_map = {"webm": "webm", "wav": "wav", "mp3": "mp3", "mp4": "mp4", "ogg": "ogg", "flac": "flac"}
        media_format = media_format_map.get(ext, "webm")

        # Start transcription job
        transcribe_kwargs = {
            "TranscriptionJobName": job_id,
            "Media": {"MediaFileUri": f"s3://{bucket}/{s3_key}"},
            "MediaFormat": media_format,
            "IdentifyLanguage": True,
        }
        if config.transcribe_vocabulary_name:
            transcribe_kwargs["LanguageIdSettings"] = {
                "en-GB": {"VocabularyName": config.transcribe_vocabulary_name},
            }
        transcribe.start_transcription_job(**transcribe_kwargs)

        # Poll until complete (typically 5-15s)
        for _ in range(60):
            resp = transcribe.get_transcription_job(TranscriptionJobName=job_id)
            status = resp["TranscriptionJob"]["TranscriptionJobStatus"]
            if status == "COMPLETED":
                break
            if status == "FAILED":
                reason = resp["TranscriptionJob"].get("FailureReason", "unknown")
                raise HTTPException(status_code=500, detail=f"Transcription failed: {reason}")
            import asyncio

            await asyncio.sleep(1)
        else:
            raise HTTPException(status_code=504, detail="Transcription timed out")

        # Fetch transcript from output URI
        import urllib.request

        transcript_uri = resp["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        with urllib.request.urlopen(transcript_uri) as r:
            transcript_data = json.loads(r.read())

        text = transcript_data["results"]["transcripts"][0]["transcript"]
        return {"text": text}

    finally:
        # Cleanup: delete temp S3 file and transcription job
        try:
            s3.delete_object(Bucket=bucket, Key=s3_key)
        except Exception:
            log.warning("Failed to delete temp S3 object %s/%s", bucket, s3_key)
        try:
            transcribe.delete_transcription_job(TranscriptionJobName=job_id)
        except Exception:
            log.warning("Failed to delete transcription job %s", job_id)


@app.post("/v1/audio/speech")
async def text_to_speech(
    request: SpeechRequest,
    _auth: None = Depends(_check_auth),
    config: RAGConfig = Depends(_get_config),
):
    """OpenAI-compatible TTS endpoint backed by Amazon Polly."""
    if config.llm_provider == "model_studio":
        raise HTTPException(status_code=501, detail="Text-to-speech not yet implemented for Model Studio provider")
    import boto3

    voice_id = _POLLY_VOICE_MAP.get(request.voice, "Arthur")
    engine = "standard" if voice_id in _POLLY_STANDARD_ONLY else "neural"

    polly = boto3.client("polly", region_name=config.bedrock_region)
    response = polly.synthesize_speech(
        Text=request.input,
        OutputFormat="mp3",
        VoiceId=voice_id,
        Engine=engine,
    )

    return Response(
        content=response["AudioStream"].read(),
        media_type="audio/mpeg",
    )


def main() -> None:
    """CLI entry point: launch the API server."""
    import uvicorn

    config = _get_config()
    log.info("Starting RAG API on port %d", config.api_port)
    uvicorn.run(app, host="0.0.0.0", port=config.api_port)


if __name__ == "__main__":
    main()
