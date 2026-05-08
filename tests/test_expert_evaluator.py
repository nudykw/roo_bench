"""Diagnostic test for ExpertEvaluator.

Sends a real large-size request to Ollama using the same code path as the
expert evaluation pipeline. Prints a detailed analysis of every API response field.

Usage:
    python test_expert_evaluator.py
    python test_expert_evaluator.py --ollama-url http://aorus-cachyos-server:11434
    python test_expert_evaluator.py --ollama-url http://aorus-cachyos-server:11434 --model gemma4:latest
"""

import sys
import logging

# ── Full debug logging identical to production ─────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_expert_evaluator")

import requests  # noqa: E402  (after logging setup so urllib3 debug is visible)

from api.local_client import LocalApiClient
from benchmark.expert_evaluator import ExpertEvaluator
from benchmark.expert_evaluator_types import ExpertEvaluationEntry
from benchmark.result import BenchmarkMetrics

# ── CLI argument parsing (mirrors main.py pattern) ─────────────────────────────
OLLAMA_URL = "http://aorus-cachyos-server:11434"
MODEL = "gemma4:latest"

_argv = sys.argv[1:]
i = 0
while i < len(_argv):
    arg = _argv[i]
    if arg in ("--ollama-url",) and i + 1 < len(_argv):
        OLLAMA_URL = _argv[i + 1]
        i += 2
    elif arg.startswith("--ollama-url="):
        OLLAMA_URL = arg[len("--ollama-url="):]
        i += 1
    elif arg in ("--model",) and i + 1 < len(_argv):
        MODEL = _argv[i + 1]
        i += 2
    elif arg.startswith("--model="):
        MODEL = arg[len("--model="):]
        i += 1
    else:
        i += 1

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


# ── Helpers ────────────────────────────────────────────────────────────────────
def header(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}")


# ── Phase 1: Inspect prompt building (no network call) ─────────────────────────
def phase1_inspect_prompt() -> tuple:
    """Build and log the evaluation prompt exactly as ExpertEvaluator._evaluate_single does."""
    header("PHASE 1 — Prompt building (no API call)")

    client    = LocalApiClient(base_url=OLLAMA_URL, timeout=120)
    evaluator = ExpertEvaluator(ollama_client=client, expert_model_name=MODEL)

    print(f"  client.base_url          : {client.base_url}")
    print(f"  client.headers           : {client.headers!r}")
    print(f"  evaluator.expert_model   : {evaluator.expert_model_name}")
    print(f"  prompts top-level keys   : {list(evaluator.prompts.keys())}")
    print(f"  prompts['expert'] keys   : {list(evaluator.prompts.get('expert', {}).keys())}")

    entry = ExpertEvaluationEntry(
        model_name  = "llama3.2:3b",
        ctx         = 32768,
        temperature = 0.7,
        prompt_id   = "architect_001",
        prompt_name = "System Design: Multi-tenant SaaS Auth",
        mode        = "architect",
        chain_id    = None,
        chain_name  = None,
        response    = LARGE_RESPONSE,
        avg_tps     = 42.5,
        metrics_ref = None,
        chain_context = {},
    )

    print(f"\n  entry.mode               : {entry.mode!r}")
    print(f"  entry.response length    : {len(entry.response)} chars")
    print(f"  entry.response[:4000] len: {min(len(entry.response), 4000)} chars")

    # ── Replicate _evaluate_single logic verbatim ──────────────────────────────
    context = (
        f"ctx={entry.ctx}, temp={entry.temperature}, "
        f"mode={entry.mode or 'default'}, prompt={entry.prompt_id}"
    )
    template = evaluator._get_prompt_template(entry.mode)
    template_vars = {
        "context":            context,
        "response":           entry.response[:4000],
        "architect_response": "N/A (independent prompt)",
        "code_response":      "N/A (independent prompt)",
    }
    if entry.chain_context:
        template_vars.update(entry.chain_context)

    try:
        eval_prompt = template.format(**template_vars)
    except KeyError as exc:
        print(f"  ⚠️  KeyError filling template: {exc} — falling back to minimal format")
        eval_prompt = template.format(context=context, response=entry.response[:4000])

    print(f"\n  Template for mode='architect':\n  {template!r}")
    print(f"\n  Final prompt length      : {len(eval_prompt)} chars")
    print(f"\n  ── Prompt first 600 chars ──")
    print(eval_prompt[:600])
    print(f"  ── Prompt last 200 chars ──")
    print(eval_prompt[-200:])

    return evaluator, entry, eval_prompt


# ── Phase 2: Raw requests.post — maximum visibility into API response ───────────
def phase2_raw_api_call(eval_prompt: str) -> dict | None:
    """Send the same payload ExpertEvaluator._call_expert_api would send,
    but intercept every byte of the response for analysis."""
    header("PHASE 2 — Raw requests.post (identical payload to _call_expert_api)")

    # Mirrors _call_expert_api payload exactly: no num_predict limit, model stops at EOS
    payload = {
        "model":  MODEL,
        "prompt": eval_prompt,
        "stream": False,
        "think":  False,
        "options": {
            "temperature": 0.1,
        },
    }
    headers_used = {}  # LocalApiClient.headers == {}

    print(f"  URL                      : {OLLAMA_URL}/api/generate")
    print(f"  payload['model']         : {payload['model']!r}")
    print(f"  payload['stream']        : {payload['stream']}")
    print(f"  payload['think']         : {payload['think']}")
    print(f"  payload['options']       : {payload['options']}")
    print(f"  prompt length            : {len(payload['prompt'])} chars")
    print(f"  headers sent             : {headers_used!r}")
    print(f"\n  ⏳ Sending request (timeout=120s)…")

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            headers=headers_used,
            timeout=120,
        )

        print(f"\n  ── HTTP Response ──")
        print(f"  status_code              : {response.status_code}")
        print(f"  Content-Type             : {response.headers.get('content-type', 'N/A')}")
        print(f"  content-length header    : {response.headers.get('content-length', '<not set>')}")
        print(f"  actual body size         : {len(response.content)} bytes")
        print(f"\n  raw body (first 1 000 bytes): {response.content[:1000]!r}")

        if response.status_code != 200:
            print(f"\n  ❌ Non-200 status — full body: {response.text[:500]!r}")
            return None

        try:
            data = response.json()
        except Exception as exc:
            print(f"\n  ❌ JSON parse failed: {exc}")
            print(f"  raw text: {response.text[:500]!r}")
            return None

        print(f"\n  ── Parsed JSON fields ──")
        print(f"  keys present             : {list(data.keys())}")

        response_field = data.get("response", "<<MISSING>>")
        print(f"\n  'response' type          : {type(response_field).__name__}")
        print(f"  'response' value (repr)  : {response_field!r}")
        print(f"  'response' length        : {len(response_field) if isinstance(response_field, str) else 'N/A'}")

        print(f"\n  'done'                   : {data.get('done')}")
        print(f"  'done_reason'            : {data.get('done_reason')!r}")
        print(f"  'model'                  : {data.get('model')!r}")
        print(f"  'eval_count'             : {data.get('eval_count')}")
        print(f"  'prompt_eval_count'      : {data.get('prompt_eval_count')}")
        td = data.get("total_duration", 0)
        ld = data.get("load_duration", 0)
        pd = data.get("prompt_eval_duration", 0)
        ed = data.get("eval_duration", 0)
        print(f"  'total_duration'         : {td / 1e9:.3f}s")
        print(f"  'load_duration'          : {ld / 1e9:.3f}s")
        print(f"  'prompt_eval_duration'   : {pd / 1e9:.3f}s")
        print(f"  'eval_duration'          : {ed / 1e9:.3f}s")

        # ── Parse score exactly as ExpertEvaluator._parse_score ────────────────
        parsed_score = ExpertEvaluator._parse_score(response_field if isinstance(response_field, str) else "5")
        print(f"\n  ── Score parsing ──")
        print(f"  ExpertEvaluator._parse_score({response_field!r}) → {parsed_score}")
        if not isinstance(response_field, str) or response_field.strip() == "":
            print(f"  ⚠️  WARNING: 'response' field is empty/missing — _parse_score will return default 5.0!")

        return data

    except requests.exceptions.Timeout:
        print(f"\n  ❌ TIMEOUT after 120 s")
    except requests.exceptions.ConnectionError as exc:
        print(f"\n  ❌ CONNECTION ERROR: {exc}")
    except Exception as exc:
        print(f"\n  ❌ UNEXPECTED ERROR: {exc}")
        import traceback
        traceback.print_exc()

    return None


# ── Phase 3: Full evaluate_batch() path with a real BenchmarkMetrics ref ───────
def phase3_evaluate_batch(evaluator: ExpertEvaluator, entry: ExpertEvaluationEntry) -> float | None:
    """Run the actual evaluate_batch() code that production uses."""
    header("PHASE 3 — evaluate_batch() with metrics_ref (production path)")

    metrics = BenchmarkMetrics(
        ctx         = entry.ctx,
        temperature = entry.temperature,
        avg_tps     = entry.avg_tps,
        min_tps     = 38.0,
        max_tps     = 47.0,
        std_dev     = 2.0,
        mode        = entry.mode,
        prompt_id   = entry.prompt_id,
        prompt_name = entry.prompt_name,
    )
    entry.metrics_ref = metrics

    print(f"  metrics.expert_score     : {metrics.expert_score}  (before)")
    print(f"  entry.metrics_ref is None: {entry.metrics_ref is None}")
    print(f"  Calling evaluator.evaluate_batch([entry])…")

    evaluator.evaluate_batch([entry])

    print(f"\n  metrics.expert_score     : {metrics.expert_score}  (after)")
    print(f"  type(expert_score)       : {type(metrics.expert_score).__name__}")
    return metrics.expert_score


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"OLLAMA_URL       : {OLLAMA_URL}")
    print(f"MODEL            : {MODEL}")
    print(f"LARGE_RESPONSE   : {len(LARGE_RESPONSE)} chars")

    try:
        evaluator, entry, eval_prompt = phase1_inspect_prompt()
        raw_data                      = phase2_raw_api_call(eval_prompt)
        final_score                   = phase3_evaluate_batch(evaluator, entry)

        header("SUMMARY")
        print(f"  eval_prompt length       : {len(eval_prompt)} chars")
        if raw_data:
            resp_field = raw_data.get("response", "<<MISSING>>")
            print(f"  raw 'response' field     : {resp_field!r}")
            print(f"  raw eval_count           : {raw_data.get('eval_count')}")
        else:
            print(f"  raw API call             : FAILED (see PHASE 2 output)")
        print(f"  evaluate_batch score     : {final_score}")

        if raw_data and raw_data.get("response", ""):
            print(f"\n  ✅ API returned a non-empty 'response' — score parsing should succeed.")
        elif raw_data:
            print()
            print("  ⚠️  DIAGNOSIS: The API returned an EMPTY 'response' field.")
            print("     _parse_score('') finds no match → returns default 5.0.")
            print("     Root cause candidates:")
            print("     1. num_predict too low — model cannot emit even one token.")
            print("     2. Model is a thinking model and 'think: false' is not propagated.")
            print("     3. Prompt is too long and model truncates before the score token.")
            print("     4. Ollama version bug with non-streaming + very short num_predict.")

    except Exception as exc:
        print(f"\nFATAL: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
