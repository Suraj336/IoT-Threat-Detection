"""
Three roles, deliberately mirroring the privacy-by-design theme of the whole
project — the dashboard itself practices data minimization by role:

  - admin:   full access, including per-home identifiable alerts and user
             management.
  - analyst: sees alerts and per-home status for triage, cannot manage users.
  - viewer:  sees only aggregate charts (training curve, alert counts by
             severity) — never a specific home's identity or raw traffic.
"""
import functools
import json
import os
import time

from flask import abort, redirect, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

USERS_PATH = os.path.join(os.path.dirname(__file__), "users.json")
ROLES = ("admin", "analyst", "viewer")

_login_attempts = {}
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


def _load_users():
    if not os.path.exists(USERS_PATH):
        return {}
    with open(USERS_PATH) as f:
        return json.load(f)


def _save_users(users):
    with open(USERS_PATH, "w") as f:
        json.dump(users, f, indent=2)


def bootstrap_default_users():
    """Creates users.json with three demo accounts on first run. Change these
    passwords immediately (see README) -- they are intentionally simple
    defaults for local evaluation only.
    """
    if os.path.exists(USERS_PATH):
        return
    users = {
        "admin": {"password_hash": generate_password_hash("ChangeMe-Admin1!"), "role": "admin"},
        "analyst": {"password_hash": generate_password_hash("ChangeMe-Analyst1!"), "role": "analyst"},
        "viewer": {"password_hash": generate_password_hash("ChangeMe-Viewer1!"), "role": "viewer"},
    }
    _save_users(users)


def is_locked_out(username):
    entry = _login_attempts.get(username)
    if not entry:
        return False
    count, last_attempt = entry
    if count >= MAX_ATTEMPTS and (time.time() - last_attempt) < LOCKOUT_SECONDS:
        return True
    if (time.time() - last_attempt) >= LOCKOUT_SECONDS:
        _login_attempts.pop(username, None)
    return False


def register_failed_attempt(username):
    count, _ = _login_attempts.get(username, (0, 0))
    _login_attempts[username] = (count + 1, time.time())


def clear_attempts(username):
    _login_attempts.pop(username, None)


def verify_login(username, password):
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    if not check_password_hash(user["password_hash"], password):
        return None
    return {"username": username, "role": user["role"]}


def create_user(username, password, role):
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    users = _load_users()
    users[username] = {"password_hash": generate_password_hash(password), "role": role}
    _save_users(users)


def delete_user(username):
    users = _load_users()
    users.pop(username, None)
    _save_users(users)


def list_users():
    users = _load_users()
    return [{"username": u, "role": v["role"]} for u, v in users.items()]


def current_user():
    if "username" not in session:
        return None
    return {"username": session["username"], "role": session.get("role")}


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def roles_required(*allowed_roles):
    def decorator(view):
        @functools.wraps(view)
        def wrapped(*args, **kwargs):
            if "username" not in session:
                return redirect(url_for("login"))
            if session.get("role") not in allowed_roles:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator
