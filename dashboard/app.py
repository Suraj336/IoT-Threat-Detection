import os

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf

import auth

ROOT = os.path.join(os.path.dirname(__file__), "..")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("DASHBOARD_SECRET_KEY", "dev-key-change-me")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Set to True once served over HTTPS (see dashboard/README.md). Left False so
# local http:// development isn't silently broken.
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("DASHBOARD_SECURE_COOKIES", "false") == "true"

csrf = CSRFProtect(app)
auth.bootstrap_default_users()


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": generate_csrf}


# ---------------------------------------------------------------- data access

def _load_metrics():
    path = os.path.join(ROOT, "models", "metrics.csv")
    if not os.path.exists(path):
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _load_alerts():
    path = os.path.join(ROOT, "data", "alerts.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "timestamp", "home_id", "predicted_class", "true_class",
            "confidence", "severity", "correct",
        ])
    return pd.read_csv(path)


def _load_client_summary():
    clients_dir = os.path.join(ROOT, "data", "clients")
    if not os.path.isdir(clients_dir):
        return []
    summary = []
    for f in sorted(os.listdir(clients_dir)):
        if not f.startswith("client_"):
            continue
        client_id = int(f.split("_")[1].split(".")[0])
        df = pd.read_csv(os.path.join(clients_dir, f))
        dist = df["label"].value_counts(normalize=True).round(3).to_dict()
        summary.append({
            "home_id": f"home-{client_id:02d}",
            "num_samples": len(df),
            "class_distribution": dist,
            "status": "online",
        })
    return summary


# --------------------------------------------------------------------- pages

@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("dashboard"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if auth.is_locked_out(username):
            error = "Too many failed attempts. Try again in a minute."
        else:
            user = auth.verify_login(username, password)
            if user:
                auth.clear_attempts(username)
                session.clear()
                session["username"] = user["username"]
                session["role"] = user["role"]
                return redirect(url_for("dashboard"))
            auth.register_failed_attempt(username)
            error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@auth.login_required
def dashboard():
    return render_template("dashboard.html", user=auth.current_user())


@app.route("/admin/users")
@auth.roles_required("admin")
def admin_users():
    return render_template("admin_users.html", user=auth.current_user(),
                            users=auth.list_users(), roles=auth.ROLES)


@app.route("/admin/users/create", methods=["POST"])
@auth.roles_required("admin")
def admin_create_user():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    role = request.form.get("role", "")
    if username and password and role in auth.ROLES:
        auth.create_user(username, password, role)
    return redirect(url_for("admin_users"))


@app.route("/admin/users/<username>/delete", methods=["POST"])
@auth.roles_required("admin")
def admin_delete_user(username):
    if username != session.get("username"):  # can't delete yourself mid-session
        auth.delete_user(username)
    return redirect(url_for("admin_users"))


# ---------------------------------------------------------------------- API
# All roles can see the training curve -- it's aggregate, not per-home.

@app.route("/api/metrics")
@auth.login_required
def api_metrics():
    return jsonify(_load_metrics())


@app.route("/api/alerts")
@auth.login_required
def api_alerts():
    df = _load_alerts()
    role = session.get("role")

    if role == "viewer":
        # Data minimization by role: viewers get counts only, never which
        # home an alert came from or its confidence/correctness detail.
        counts = df["severity"].value_counts().to_dict() if not df.empty else {}
        return jsonify({"severity_counts": counts})

    # admin + analyst see full alert detail
    records = df.head(200).to_dict(orient="records")  # cap payload size
    return jsonify({"alerts": records})


@app.route("/api/clients")
@auth.roles_required("admin", "analyst")
def api_clients():
    return jsonify(_load_client_summary())


@app.route("/api/whoami")
@auth.login_required
def api_whoami():
    return jsonify(auth.current_user())


if __name__ == "__main__":
    # debug=False by default even locally -- flip only for active development,
    # never in anything reachable outside your machine.
    app.run(host="127.0.0.1", port=5050, debug=False)
