import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI  # noqa: E402
from backend.api.main import app as backend_app  # noqa: E402

# Vercel routes /api/* to this file (path forwarded as-is), so FastAPI sees
# /api/regime/current. Mount backend_app at /api so it receives /regime/current.
app = FastAPI()
app.mount("/api", backend_app)

handler = app
