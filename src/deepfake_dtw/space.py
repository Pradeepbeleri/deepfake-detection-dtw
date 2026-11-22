"""Single-operator Hugging Face Space with guarded access to the local API."""
import os
import re
import secrets
from pathlib import Path

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.staticfiles import StaticFiles

from .api import create_app


def create_space():
    host = os.environ['SPACE_HOST']
    if not re.fullmatch(r'[a-zA-Z0-9-]+\.hf\.space', host):
        raise ValueError('Configure the canonical Hugging Face Space hostname')
    owner = os.environ['DEEPFAKE_DEMO_USER'].strip().casefold()
    session_secret = os.environ['DEEPFAKE_SESSION_SECRET']
    if not owner or len(session_secret) < 32:
        raise ValueError('Configure an approved Hugging Face username and session secret')

    callback = f'https://{host}/login/callback'
    oauth = OAuth()
    huggingface = oauth.register(
        name='huggingface',
        client_id=os.environ['OAUTH_CLIENT_ID'],
        client_secret=os.environ['OAUTH_CLIENT_SECRET'],
        server_metadata_url='https://huggingface.co/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid profile', 'code_challenge_method': 'S256'},
    )

    # The existing API remains an in-process, bearer-protected ASGI app.
    token = secrets.token_urlsafe(32)
    os.environ['DEEPFAKE_API_TOKEN'] = token
    internal = create_app()
    app = FastAPI(
        docs_url=None, redoc_url=None, openapi_url=None,
        lifespan=internal.router.lifespan_context,
    )
    app.add_middleware(SessionMiddleware, secret_key=session_secret,
                       session_cookie='__Host-deepfake-session', https_only=True, max_age=3600)

    @app.get('/login')
    async def login(request: Request):
        request.session.clear()
        return await huggingface.authorize_redirect(request, callback)

    @app.get('/login/callback')
    async def login_callback(request: Request):
        try:
            credentials = await huggingface.authorize_access_token(request)
        except OAuthError:
            return JSONResponse({'detail': 'Sign-in failed. Reopen the Space to try again.'}, status_code=401)
        profile = credentials.get('userinfo', {})
        name = profile.get('preferred_username')
        request.session.clear()
        if not isinstance(name, str) or name.casefold() != owner:
            return JSONResponse({'detail': 'This research demo is restricted to its operator.'}, status_code=403)
        request.session['operator'] = owner
        return RedirectResponse('/', status_code=303)

    async def gated_api(scope, receive, send):
        request = Request(scope)
        if request.session.get('operator') != owner:
            await JSONResponse({'detail': 'Sign in to the demo first.'}, status_code=401)(scope, receive, send)
            return
        # A custom header keeps cross-site forms from submitting or deleting jobs.
        if request.method in ('POST', 'DELETE') and request.headers.get('x-demo-request') != '1':
            await JSONResponse({'detail': 'Invalid demo request.'}, status_code=403)(scope, receive, send)
            return
        if scope['path'].removeprefix(scope['root_path']) == '/health':
            await JSONResponse({
                'service': 'available', 'model': 'configured_unverified',
                'access': 'hosted', 'note': 'Model verified on startup.',
            }, headers={'Cache-Control': 'no-store'})(scope, receive, send)
            return
        headers = [(k, v) for k, v in scope['headers'] if k.lower() != b'authorization']
        headers.append((b'authorization', f'Bearer {token}'.encode()))
        await internal({**scope, 'client': ('127.0.0.1', 0), 'headers': headers}, receive, send)

    @app.get('/')
    async def index(request: Request):
        if request.session.get('operator') != owner:
            return RedirectResponse('/login')
        # The frontend assets contain no credentials or private model weights.
        return FileResponse(frontend / 'index.html', headers={'Cache-Control': 'no-store'})

    frontend = Path(os.environ.get('DEEPFAKE_FRONTEND_DIR', '/app/frontend/dist'))
    app.mount('/api', gated_api)
    app.mount('/assets', StaticFiles(directory=frontend / 'assets'))
    return app
