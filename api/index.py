# api/index.py
import sys
from pathlib import Path

# Add project root to Python path so backend.* is importable in Vercel's serverless context
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api.main import app  # noqa: E402

# Vercel Python runtime detects the ASGI `app` object automatically
handler = app
