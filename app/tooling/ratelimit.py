from slowapi import Limiter
from slowapi.util import get_remote_address

from app.settings import settings

# Default: per-IP rate limiting
limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)
