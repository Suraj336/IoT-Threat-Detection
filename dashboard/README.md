# Dashboard

A secure, role-based web dashboard for the federated IoT threat-detection
system. It reads directly from the pipeline's own outputs — `models/metrics.csv`
(training curve) and a trained `models/global_model.pt` (used to generate real
alerts) — so what you see here is not mock data.

## What it shows

| View | admin | analyst | viewer |
|---|---|---|---|
| Training accuracy / F1 curve (aggregate) | ✅ | ✅ | ✅ |
| Alert counts by severity | ✅ | ✅ | ✅ |
| Individual alerts (which home, confidence, timestamp) | ✅ | ✅ | ❌ |
| Per-home status grid | ✅ | ✅ | ❌ |
| User management | ✅ | ❌ | ❌ |

The viewer role deliberately gets *less* data, not just a read-only version
of the same page — that data-minimization-by-role is the same principle the
federated learning system itself is built on.

## Setup

```bash
cd dashboard
python3 -m venv venv        # separate venv from the FL pipeline is fine, or reuse the root one
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# generate a real secret key and paste it into .env:
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Load the `.env` file (or export the two variables manually) before running:

```bash
export $(cat .env | xargs)
```

## Reproduction steps (full pipeline → dashboard)

```bash
# from the project root, with the main venv active
python data/prepare_data.py --num_clients 10 --non_iid_alpha 0.3
python src/server.py --rounds 20 --num_clients 10   # now also saves models/global_model.pt
python dashboard/generate_alerts.py                  # runs the trained model, writes data/alerts.csv

cd dashboard
python app.py
```

Open **http://127.0.0.1:5050** and log in.

## Default accounts

Created automatically on first run (`dashboard/users.json`). **Change these
before doing anything beyond local evaluation:**

| Username | Password | Role |
|---|---|---|
| `uman` | `uman1234` | admin |
| `suraj` | `suraj123` | analyst |
| `saugat` | `saugat123` | viewer |

To change a password: log in as admin → Manage users → remove the account →
re-create it with a new password. (There's no in-place password change yet —
see the security notes below.)

## Security measures implemented

- Passwords are hashed with Werkzeug's PBKDF2 implementation, never stored
  or logged in plaintext.
- Session cookies are `HttpOnly` and `SameSite=Lax` by default; `Secure` is
  a one-line env-var flip once you're behind HTTPS.
- CSRF tokens (Flask-WTF) on every form (login, create user, delete user).
- Role-based access control on every route and API endpoint via
  `@auth.roles_required(...)`, not just hidden UI elements — a viewer
  account calling `/api/clients` directly gets a 403, not just a missing
  button.
- A simple login-attempt lockout (5 failed attempts → 60s lockout per
  username) to blunt naive brute-forcing.
- `debug=False` by default in `app.py`, so stack traces are never exposed.

## Hardening for production

This is built to be honest about its current scope — a local/demo-grade
dashboard, not a hardened internet-facing service. Before deploying beyond
your own machine:

- **Move users.json to a real database** with proper migrations, and add
  password-change/reset flows instead of delete-and-recreate.
- **Replace the in-memory login-attempt tracker** with something that
  survives process restarts and works across multiple app instances
  (e.g. Flask-Limiter backed by Redis).
- **Add structured audit logging** for login attempts and admin actions
  (who created/deleted which user, when).
- **Run behind a WSGI server** (gunicorn/uwsgi), not Flask's built-in dev
  server, and set `DASHBOARD_SECRET_KEY` from a secrets manager rather than
  a `.env` file.

## Troubleshooting

- **"No training runs yet" on the chart** → run `python src/server.py` from
  the project root first.
- **"No alerts yet"** → run `python dashboard/generate_alerts.py` after a
  training run has produced `models/global_model.pt`.
- **403 Forbidden** on a page → that's role-based access control working as
  intended; log in with an account that has the right role.
