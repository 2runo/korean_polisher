from korean_polisher.server import app
from korean_polisher.utils import get_env

port = get_env('PORT', 8000)
app.run(host="0.0.0.0", port=port)
