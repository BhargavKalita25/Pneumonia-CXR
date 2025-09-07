import time
import os
import threading
from fastapi import HTTPException, UploadFile

# Thread-safe in-memory rate limiter with periodic cleanup
_RATE_LOCK = threading.Lock()
RATE = {}
WINDOW = 10  # seconds
ALLOW = 60   # max requests per window
LAST_CLEANUP = time.time()

def rate_limiter(key: str):
    """Token-bucket style rate limiter with automatic stale-entry pruning."""
    global LAST_CLEANUP
    if not key:
        key = "unknown_client"

    now = time.time()
    with _RATE_LOCK:
        # Periodic eviction of old keys (every 60s)
        if now - LAST_CLEANUP > 60:
            stale_keys = [k for k, timestamps in RATE.items() if not timestamps or now - timestamps[-1] >= WINDOW]
            for k in stale_keys:
                del RATE[k]
            LAST_CLEANUP = now

        bucket = [t for t in RATE.get(key, []) if now - t < WINDOW]
        if len(bucket) >= ALLOW:
            raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
        bucket.append(now)
        RATE[key] = bucket

def validate_upload(
    filename: str,
    content: bytes,
    max_mb: int = 15,
    allowed_ext=(".png", ".jpg", ".jpeg", ".dcm")
):
    """Validate file extension, size limit, and header magic bytes."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(allowed_ext)}"
        )

    # Size validation
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            status_code=413,
            detail=f"Payload exceeds maximum allowed size of {max_mb} MB (received {size_mb:.2f} MB)."
        )

    # Magic byte integrity checks
    if ext == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(status_code=400, detail="Invalid PNG file header.")
    elif ext in (".jpg", ".jpeg") and not content.startswith(b"\xff\xd8\xff"):
        raise HTTPException(status_code=400, detail="Invalid JPEG file header.")
    elif ext == ".dcm":
        # DICOM preamble is 128 bytes followed by magic word 'DICM'
        if len(content) > 132 and content[128:132] != b"DICM":
            # Some DICOMs omit the 128-byte preamble, allow fallback but check minimum length
            if len(content) < 256:
                raise HTTPException(status_code=400, detail="Corrupted or invalid DICOM file.")

    return True
