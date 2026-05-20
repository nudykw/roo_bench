"""Test data and constants for diagnostic tests.

Contains large synthetic data used by test_expert_evaluator.py
to test the ExpertEvaluator with realistically-sized payloads.
"""

# ── Large synthetic benchmark response (~4 000+ chars) ─────────────────────────
# Simulates what a tested model would return for an architect-mode prompt so
# the expert evaluator receives a realistically-sized payload.
LARGE_RESPONSE = """\
# FastAPI Multi-Tenant SaaS Authentication System — Architecture Design

## 1. Executive Summary

This document presents a production-ready architecture for a multi-tenant SaaS
platform supporting thousands of isolated tenants, role-based access control
(RBAC), stateless JWT authentication with refresh-token rotation, and a
PostgreSQL multi-schema data isolation strategy.

## 2. Core Components

### 2.1 Authentication Layer
- **JWT access tokens** (HS256, 30-minute TTL) issued on successful login.
- **Refresh tokens** (SHA-256 opaque, 7-day TTL) stored in Redis with
  automatic rotation on use (sliding window).
- **bcrypt** password hashing (cost factor 12 minimum, configurable via env).
- OAuth2 social login (Google, GitHub) via `authlib` — future phase.

### 2.2 Multi-Tenancy Strategy
- **Schema-per-tenant** PostgreSQL isolation: each tenant owns a dedicated
  `pg_schema` prefixed with `t_{tenant_slug}`.
- Schema switching via `SET search_path = t_{slug}, public` at the session
  level inside a FastAPI dependency, transparent to business logic.
- A **shared** `public` schema holds the tenant registry and cross-tenant
  tables (billing, global audit).
- `SQLAlchemy` async sessions with per-tenant connection pools (max 20
  connections each), capped at 1 000 total via a semaphore.

### 2.3 RBAC System
- Five built-in roles per tenant: `superadmin`, `admin`, `manager`, `user`,
  `readonly`.
- Custom roles created at runtime; permissions stored as a JSONB array
  of `"resource:action"` strings (e.g. `"invoices:read"`).
- FastAPI `Depends` decorator injects permission checks at the route level:

```python
@router.get("/invoices", dependencies=[Depends(require("invoices:read"))])
async def list_invoices(db: AsyncSession = Depends(get_db)):
    ...
```

## 3. Database Schema (PostgreSQL 16)

```sql
-- Shared registry
CREATE TABLE public.tenants (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug        VARCHAR(50)  UNIQUE NOT NULL,
    name        VARCHAR(200) NOT NULL,
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Per-tenant users (schema substituted at migration time)
CREATE TABLE {schema}.users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role_id         UUID REFERENCES {schema}.roles(id) ON DELETE SET NULL,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT now(),
    last_login      TIMESTAMPTZ
);

CREATE TABLE {schema}.roles (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        VARCHAR(50) UNIQUE NOT NULL,
    permissions JSONB NOT NULL DEFAULT '[]'
);

CREATE TABLE {schema}.refresh_tokens (
    token_hash  CHAR(64) PRIMARY KEY,
    user_id     UUID REFERENCES {schema}.users(id) ON DELETE CASCADE,
    expires_at  TIMESTAMPTZ NOT NULL,
    rotated_at  TIMESTAMPTZ,
    revoked     BOOLEAN DEFAULT FALSE
);

CREATE TABLE {schema}.audit_log (
    id         BIGSERIAL PRIMARY KEY,
    user_id    UUID REFERENCES {schema}.users(id),
    action     VARCHAR(100) NOT NULL,
    resource   VARCHAR(200),
    metadata   JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

## 4. FastAPI Implementation Sketch

```python
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import jwt, bcrypt, hashlib, secrets
from datetime import datetime, timedelta, timezone

app = FastAPI(title="Multi-Tenant Auth API", version="1.0.0")
security = HTTPBearer()

JWT_SECRET    = os.getenv("JWT_SECRET")   # 256-bit min
JWT_ALG       = "HS256"
ACCESS_EXPIRE = timedelta(minutes=30)
REFRESH_DAYS  = 7

# ── Token helpers ──────────────────────────────────────────────────────────────
def create_access_token(user_id: str, tenant: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": user_id, "tenant": tenant,
         "iat": now, "exp": now + ACCESS_EXPIRE},
        JWT_SECRET, algorithm=JWT_ALG,
    )

def create_refresh_token(user_id: str, tenant: str, db_session) -> str:
    raw   = secrets.token_hex(32)
    h     = hashlib.sha256(raw.encode()).hexdigest()
    token = RefreshToken(token_hash=h, user_id=user_id,
                         expires_at=datetime.now(timezone.utc) +
                         timedelta(days=REFRESH_DAYS))
    db_session.add(token)
    return raw  # returned to client, hash stored

# ── Tenant resolution ──────────────────────────────────────────────────────────
async def resolve_tenant(request: Request, db: AsyncSession = Depends(get_db)):
    host = request.headers.get("host", "")
    slug = host.split(".")[0]
    result = await db.execute(select(Tenant).where(Tenant.slug == slug,
                                                    Tenant.is_active == True))
    tenant = result.scalar_one_or_none()
    if not tenant:
        raise HTTPException(404, "Tenant not found")
    await db.execute(text(f"SET search_path = t_{slug}, public"))
    return slug

# ── Auth dependency ────────────────────────────────────────────────────────────
async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    tenant: str = Depends(resolve_tenant),
    db: AsyncSession = Depends(get_db),
) -> User:
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

    if payload.get("tenant") != tenant:
        raise HTTPException(403, "Token/tenant mismatch")

    user = await db.get(User, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(401, "User inactive or not found")
    return user

# ── Login endpoint ─────────────────────────────────────────────────────────────
@app.post("/auth/login")
async def login(form: LoginForm, request: Request,
                tenant: str = Depends(resolve_tenant),
                db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, form.email, form.password)
    if not user:
        raise HTTPException(401, "Invalid credentials")
    access  = create_access_token(str(user.id), tenant)
    refresh = create_refresh_token(str(user.id), tenant, db)
    await db.commit()
    await audit(db, user.id, "login", ip=request.client.host)
    return {"access_token": access, "refresh_token": refresh,
            "token_type": "bearer", "expires_in": int(ACCESS_EXPIRE.total_seconds())}
```

## 5. Deployment (Docker Compose)

```yaml
version: "3.9"
services:
  api:
    image: saas-auth:latest
    env_file: .env
    depends_on: [db, redis]
    deploy:
      replicas: 3
      restart_policy: {condition: on-failure}
  db:
    image: postgres:16-alpine
    volumes: [pgdata:/var/lib/postgresql/data]
    environment: {POSTGRES_DB: saas, POSTGRES_PASSWORD: "${DB_PASS}"}
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
  nginx:
    image: nginx:alpine
    ports: ["443:443", "80:80"]
volumes:
  pgdata:
```

## 6. Security Hardening Checklist

| Control | Implementation |
|---------|----------------|
| Password hashing | bcrypt cost=12, upgrade on login if cost changed |
| JWT secret rotation | 90-day rotation, 5-minute overlap window |
| Rate limiting | 5 auth failures / min / IP via Redis sliding window |
| SQL injection | Parameterised queries only (SQLAlchemy ORM + `text()` bind params) |
| CORS | Allowlist: `*.{tenant}.example.com` per tenant |
| TLS | 1.3 minimum, HSTS max-age=63072000, includeSubDomains |
| Refresh token theft | Device fingerprint hash stored with token |
| Audit trail | All auth events (login, logout, pw-change, failed attempt) |

## 7. Performance Targets

- Login p99 < 200 ms (including bcrypt verify)
- JWT validation (in-memory) < 2 ms
- 10 000 concurrent tenants on 3 API replicas
- Connection pool: 20 per tenant, hard cap 1 000 total via semaphore

## 8. Summary

The schema-per-tenant PostgreSQL strategy provides strong data isolation without
the operational complexity of separate databases. The stateless JWT layer scales
horizontally with no shared in-process state. Refresh-token rotation limits the
blast radius of token leakage. RBAC is flexible enough to support both default
roles and customer-specific permission sets.
"""


# ── Default CLI values ─────────────────────────────────────────────────────────
DEFAULT_OLLAMA_URL = "http://aorus-cachyos-server:11434"
DEFAULT_MODEL = "gemma4:latest"


def parse_cli_args(argv: list[str] | None = None) -> tuple[str, str]:
    """Parse command-line arguments for diagnostic tests.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Tuple of (ollama_url, model).
    """
    if argv is None:
        import sys
        argv = sys.argv[1:]

    ollama_url = DEFAULT_OLLAMA_URL
    model = DEFAULT_MODEL

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--ollama-url",) and i + 1 < len(argv):
            ollama_url = argv[i + 1]
            i += 2
        elif arg.startswith("--ollama-url="):
            ollama_url = arg[len("--ollama-url="):]
            i += 1
        elif arg in ("--model",) and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
        elif arg.startswith("--model="):
            model = arg[len("--model="):]
            i += 1
        else:
            i += 1

    return ollama_url, model
