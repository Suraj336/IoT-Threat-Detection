const SEVERITY_ORDER = ["critical", "high", "medium", "info"];

async function fetchJSON(url) {
  const res = await fetch(url, { credentials: "same-origin" });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function renderMetricsChart(metrics) {
  const ctx = document.getElementById("metrics-chart");
  if (!metrics.length) {
    ctx.parentElement.innerHTML = '<div class="empty-state">No training runs yet — run `python src/server.py` to populate this chart.</div>';
    return;
  }

  const labels = metrics.map((m) => `R${m.round}`);
  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Accuracy",
          data: metrics.map((m) => m.accuracy),
          borderColor: "#E8A33D",
          backgroundColor: "rgba(232,163,61,0.08)",
          tension: 0.25,
          fill: true,
        },
        {
          label: "F1 (macro)",
          data: metrics.map((m) => m.f1_macro),
          borderColor: "#3FA796",
          backgroundColor: "rgba(63,167,150,0.08)",
          tension: 0.25,
          fill: true,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1, ticks: { color: "#8A99A8" }, grid: { color: "#26333F" } },
        x: { ticks: { color: "#8A99A8" }, grid: { display: false } },
      },
      plugins: {
        legend: { labels: { color: "#E4E9EE", font: { family: "IBM Plex Sans" } } },
      },
    },
  });

  const last = metrics[metrics.length - 1];
  document.getElementById("kpi-accuracy").textContent = (last.accuracy * 100).toFixed(1) + "%";
  document.getElementById("kpi-f1").textContent = (last.f1_macro * 100).toFixed(1) + "%";
}

function severityCounts(alerts) {
  const counts = {};
  alerts.forEach((a) => { counts[a.severity] = (counts[a.severity] || 0) + 1; });
  return counts;
}

function renderAlertsFull(alerts) {
  const container = document.getElementById("alerts-container");
  if (!alerts.length) {
    container.innerHTML = '<div class="empty-state">No alerts yet — run `python dashboard/generate_alerts.py` after training a model.</div>';
    return;
  }

  const rows = alerts.slice(0, 50).map((a) => `
    <tr>
      <td>${new Date(a.timestamp).toLocaleString()}</td>
      <td>${a.home_id}</td>
      <td>${a.predicted_class}</td>
      <td><span class="severity-chip ${a.severity}">${a.severity}</span></td>
      <td>${(a.confidence * 100).toFixed(1)}%</td>
      <td>${a.correct ? "confirmed" : "unverified"}</td>
    </tr>`).join("");

  container.innerHTML = `
    <table>
      <thead><tr><th>Time</th><th>Home</th><th>Detected as</th><th>Severity</th><th>Confidence</th><th>Status</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderAlertsAggregate(counts) {
  const container = document.getElementById("alerts-container");
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (!total) {
    container.innerHTML = '<div class="empty-state">No alerts yet.</div>';
    return;
  }
  const chips = SEVERITY_ORDER
    .filter((s) => counts[s])
    .map((s) => `<span class="severity-chip ${s}" style="margin-right: 8px; font-size: 13px; padding: 5px 10px;">${s}: ${counts[s]}</span>`)
    .join("");
  container.innerHTML = `<p style="color: var(--text-muted); font-size: 13px; margin-bottom: 12px;">
    Your role shows aggregate counts only — per-home detail is restricted to analyst and admin accounts.</p>${chips}`;
}

function renderClients(clients) {
  const grid = document.getElementById("client-grid");
  if (!grid) return;
  if (!clients.length) {
    grid.innerHTML = '<div class="empty-state">No client data yet — run `python data/prepare_data.py`.</div>';
    return;
  }
  grid.innerHTML = clients.map((c) => `
    <div class="client-node">
      <div class="home-id"><span class="status-dot"></span>${c.home_id}</div>
      <div class="sample-count">${c.num_samples} flows/round</div>
    </div>`).join("");
}

async function init() {
  const metrics = await fetchJSON("/api/metrics");
  renderMetricsChart(metrics);

  const alertsResp = await fetchJSON("/api/alerts");
  if (alertsResp.alerts) {
    renderAlertsFull(alertsResp.alerts);
    const counts = severityCounts(alertsResp.alerts);
    document.getElementById("kpi-critical").textContent = counts.critical || 0;
    document.getElementById("kpi-high").textContent = counts.high || 0;
  } else {
    renderAlertsAggregate(alertsResp.severity_counts || {});
    document.getElementById("kpi-critical").textContent = (alertsResp.severity_counts || {}).critical || 0;
    document.getElementById("kpi-high").textContent = (alertsResp.severity_counts || {}).high || 0;
  }

  if (window.CURRENT_ROLE === "admin" || window.CURRENT_ROLE === "analyst") {
    const clients = await fetchJSON("/api/clients");
    renderClients(clients);
  }
}

init().catch((err) => {
  console.error(err);
});
