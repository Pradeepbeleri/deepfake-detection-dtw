"""Single-operator hosted demo with guarded access to the local API."""
import os
import re
from pathlib import Path

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.staticfiles import StaticFiles

from .api import create_app


def create_space():
    render_host = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
    host = render_host or os.environ['SPACE_HOST']
    provider = 'google' if render_host else 'huggingface'
    suffix = 'onrender.com' if render_host else 'hf.space'
    if not re.fullmatch(r'[a-zA-Z0-9-]+\.' + re.escape(suffix), host):
        raise ValueError('Configure the canonical deployment hostname')
    owner_key = 'DEEPFAKE_DEMO_EMAIL' if render_host else 'DEEPFAKE_DEMO_USER'
    owner = os.environ[owner_key].strip().casefold()
    session_secret = os.environ['DEEPFAKE_SESSION_SECRET']
    if not owner or len(session_secret) < 32:
        raise ValueError('Configure an approved operator and session secret')

    callback = f'https://{host}/login/callback'
    oauth = OAuth()
    identity_provider = oauth.register(
        name=provider,
        client_id=os.environ['OAUTH_CLIENT_ID'],
        client_secret=os.environ['OAUTH_CLIENT_SECRET'],
        server_metadata_url=('https://accounts.google.com/.well-known/openid-configuration' if render_host
                             else 'https://huggingface.co/.well-known/openid-configuration'),
        client_kwargs={'scope': 'openid email' if render_host else 'openid profile',
                       'code_challenge_method': 'S256'},
    )

    # The internal API is reachable only through the signed-in route below.
    internal = create_app()
    app = FastAPI(
        docs_url=None, redoc_url=None, openapi_url=None,
        lifespan=internal.router.lifespan_context,
    )
    app.add_middleware(SessionMiddleware, secret_key=session_secret,
                       session_cookie='__Host-deepfake-session', https_only=True, max_age=3600)

    @app.get('/healthz')
    async def healthz():
        return {'status': 'ok'}

    @app.get('/login')
    async def login(request: Request):
        request.session.clear()
        return await identity_provider.authorize_redirect(request, callback)

    @app.get('/login/callback')
    async def login_callback(request: Request):
        try:
            credentials = await identity_provider.authorize_access_token(request)
        except OAuthError:
            return JSONResponse({'detail': 'Sign-in failed. Reopen the demo to try again.'}, status_code=401)
        profile = credentials.get('userinfo', {})
        name = profile.get('email' if render_host else 'preferred_username')
        verified = not render_host or profile.get('email_verified') is True
        request.session.clear()
        if not verified or not isinstance(name, str) or name.casefold() != owner:
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
        headers = [(k, v) for k, v in scope['headers'] if k.lower() != b'host']
        headers.append((b'host', b'localhost'))
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
