from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, List

from flask import Flask, request, redirect, url_for, render_template_string, jsonify, send_from_directory

from src.fairseq.runner import run_instance_trace

APP_TITLE = "Fair Sequence – Dashboard"
DEFAULT_RESULTS_DIR = Path("results/dashboard_runs")

# In-memory task store (fine for thesis/demo). For persistence, write task state to disk.
TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()

def create_app() -> Flask:
    app = Flask(__name__)
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def list_classes(data_root: Path) -> List[str]:
        if not data_root.exists():
            return []
        return sorted([p.name for p in data_root.iterdir() if p.is_dir()])

    def list_instances(data_root: Path, class_name: str) -> List[str]:
        p = data_root / class_name
        if not p.exists():
            return []
        inst_dirs = [d for d in p.iterdir() if d.is_dir()]
        # numeric sort when possible
        def key(x: Path):
            return int(x.name) if x.name.isdigit() else x.name
        return [d.name for d in sorted(inst_dirs, key=key)]

    def safe_path(p: str) -> Path:
        return Path(p).expanduser().resolve()

    @app.get("/")
    def index():
        data_root = request.args.get("data_root", "data")
        data_root_p = safe_path(data_root) if data_root else Path("data").resolve()

        cls = request.args.get("class", None)
        classes = list_classes(data_root_p)
        if cls is None and classes:
            cls = classes[0]
        instances = list_instances(data_root_p, cls) if cls else []
        inst = request.args.get("instance", None)
        if inst is None and instances:
            inst = instances[0]

        # Recent runs (from disk)
        recent = []
        for jf in sorted(DEFAULT_RESULTS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:15]:
            try:
                recent.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:
                continue

        html = """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>{{title}}</title>
          <style>
            body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
            .row { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-end; }
            .card { border: 1px solid #ddd; border-radius: 14px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
            .card h2 { margin: 0 0 10px 0; font-size: 18px; }
            label { display: block; font-size: 13px; color: #444; margin-bottom: 6px; }
            select, input { padding: 8px 10px; border: 1px solid #ccc; border-radius: 10px; min-width: 220px; }
            button { padding: 10px 14px; border-radius: 12px; border: 0; background: #111; color: #fff; cursor: pointer; }
            button:hover { opacity: 0.9; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border-bottom: 1px solid #eee; padding: 8px 10px; font-size: 13px; text-align: left; }
            .muted { color: #666; font-size: 13px; }
            a { color: #0b63ce; text-decoration: none; }
            a:hover { text-decoration: underline; }
          </style>
        </head>
        <body>
          <h1>{{title}}</h1>
          <p class="muted">Lancia run controllate (algoritmo, inizializzazione, time limit) e visualizza curve best-so-far.</p>

          <div class="row">
            <div class="card">
              <h2>Nuova esecuzione</h2>
              <form method="post" action="{{url_for('start_run')}}">
                <div class="row">
                  <div>
                    <label>Data root</label>
                    <input name="data_root" value="{{data_root}}" />
                  </div>

                  <div>
                    <label>Classe</label>
                    <select name="class" onchange="this.form.submit()">
                      {% for c in classes %}
                        <option value="{{c}}" {% if c==cls %}selected{% endif %}>{{c}}</option>
                      {% endfor %}
                    </select>
                  </div>

                  <div>
                    <label>Istanza</label>
                    <select name="instance">
                      {% for i in instances %}
                        <option value="{{i}}" {% if i==inst %}selected{% endif %}>{{i}}</option>
                      {% endfor %}
                    </select>
                  </div>
                </div>

                <div class="row" style="margin-top:12px;">
                  <div>
                    <label>Algoritmo</label>
                    <select name="algo">
                      {% for a in ['ls','sa','tabu','ga'] %}
                        <option value="{{a}}">{{a}}</option>
                      {% endfor %}
                    </select>
                  </div>

                  <div>
                    <label>Soluzione iniziale</label>
                    <select name="init_method">
                      {% for m in ['round_robin','random','greedy_fair'] %}
                        <option value="{{m}}">{{m}}</option>
                      {% endfor %}
                    </select>
                  </div>

                  <div>
                    <label>Time limit (sec)</label>
                    <input name="time_limit_s" type="number" value="180" min="1" step="1"/>
                  </div>

                  <div>
                    <label>Seed</label>
                    <input name="seed" type="number" value="1" min="0" step="1"/>
                  </div>

                  <div>
                    <label>Log every (sec)</label>
                    <input name="log_every_s" type="number" value="1" min="0.2" step="0.2"/>
                  </div>

                  <div>
                    <button type="submit">Avvia</button>
                  </div>
                </div>
              </form>
            </div>

            <div class="card" style="min-width:360px; flex:1;">
              <h2>Run recenti</h2>
              {% if recent %}
                <table>
                  <thead><tr>
                    <th>Quando</th><th>Algo</th><th>Init</th><th>TL</th><th>Best max-avg</th><th></th>
                  </tr></thead>
                  <tbody>
                    {% for r in recent %}
                      <tr>
                        <td>{{r.get('ended_at','')}}</td>
                        <td>{{r.get('algo','')}}</td>
                        <td>{{r.get('init','')}}</td>
                        <td>{{r.get('time_limit_s','')}}</td>
                        <td>{{"%.4f"|format(r.get('best_max_avg_completion',0.0))}}</td>
                        <td><a href="{{url_for('view_run', run_id=r.get('run_id'))}}">Apri</a></td>
                      </tr>
                    {% endfor %}
                  </tbody>
                </table>
              {% else %}
                <p class="muted">Nessuna run salvata ancora.</p>
              {% endif %}
            </div>
          </div>

          <div style="margin-top:18px;">
            <a href="{{url_for('compare')}}">Vai alla pagina “Compare”</a>
          </div>

        </body>
        </html>
        """
        return render_template_string(
            html, title=APP_TITLE, data_root=data_root, classes=classes, cls=cls, instances=instances, inst=inst, recent=recent
        )

    @app.post("/start")
    def start_run():
        data_root = request.form.get("data_root", "data")
        cls = request.form.get("class")
        inst = request.form.get("instance")
        algo = request.form.get("algo", "sa")
        init_method = request.form.get("init_method", "round_robin")
        time_limit_s = int(float(request.form.get("time_limit_s", "180")))
        seed = int(float(request.form.get("seed", "1")))
        log_every_s = float(request.form.get("log_every_s", "1"))

        data_root_p = safe_path(data_root)
        instance_dir = data_root_p / cls / inst

        run_id = uuid.uuid4().hex[:10]
        task_id = uuid.uuid4().hex

        task = {
            "task_id": task_id,
            "run_id": run_id,
            "status": "queued",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": "",
            "instance_dir": str(instance_dir),
            "algo": algo,
            "init": init_method,
            "time_limit_s": time_limit_s,
            "seed": seed,
            "log_every_s": log_every_s,
            "trace": [],
            "result": None,
            "error": None,
        }
        with TASKS_LOCK:
            TASKS[task_id] = task

        def worker():
            with TASKS_LOCK:
                TASKS[task_id]["status"] = "running"
            try:
                res, trace = run_instance_trace(
                    Path(instance_dir),
                    algo=algo,
                    time_limit_s=time_limit_s,
                    seed=seed,
                    init_method=init_method,
                    log_every_s=log_every_s,
                    verbose=False,
                )
                ended = time.strftime("%Y-%m-%d %H:%M:%S")
                payload = {
                    "run_id": run_id,
                    "ended_at": ended,
                    "instance_dir": str(instance_dir),
                    "algo": algo,
                    "init": init_method,
                    "time_limit_s": time_limit_s,
                    "seed": seed,
                    "log_every_s": log_every_s,
                    "best_obj": res.best_obj,
                    "best_max_avg_completion": res.best_max_avg,
                    "best_sum_avg_completion": res.best_sum_avg,
                    "feasible": res.feasible,
                    "violation": res.violation,
                    "n_iters": res.n_iters,
                    "elapsed_s": res.elapsed_s,
                    "trace": trace,
                }
                # persist to disk
                out = DEFAULT_RESULTS_DIR / f"run_{run_id}.json"
                out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                with TASKS_LOCK:
                    TASKS[task_id]["status"] = "done"
                    TASKS[task_id]["ended_at"] = ended
                    TASKS[task_id]["trace"] = trace
                    TASKS[task_id]["result"] = asdict(res)
            except Exception as e:
                with TASKS_LOCK:
                    TASKS[task_id]["status"] = "error"
                    TASKS[task_id]["error"] = repr(e)

        threading.Thread(target=worker, daemon=True).start()
        return redirect(url_for("task_page", task_id=task_id))

    @app.get("/task/<task_id>")
    def task_page(task_id: str):
        html = """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>Run {{task_id}}</title>
          <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
          <style>
            body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
            .card { border: 1px solid #ddd; border-radius: 14px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 14px; }
            .muted { color: #666; font-size: 13px; }
            a { color: #0b63ce; text-decoration: none; }
            a:hover { text-decoration: underline; }
            code { background:#f6f6f6; padding:2px 6px; border-radius:8px; }
          </style>
        </head>
        <body>
          <h1>Run in esecuzione</h1>
          <p><a href="{{url_for('index')}}">← Home</a></p>

          <div class="card">
            <div id="meta" class="muted">Caricamento…</div>
          </div>

          <div class="card">
            <h2>Best max-avg completion vs tempo</h2>
            <canvas id="chart" height="110"></canvas>
            <p class="muted">La curva è “best-so-far”: quando l’algoritmo trova una soluzione migliore, il valore scende.</p>
          </div>

          <div class="card">
            <h2>Esito</h2>
            <pre id="result" class="muted">In attesa…</pre>
          </div>

          <script>
            const taskId = "{{task_id}}";
            const ctx = document.getElementById('chart');
            const chart = new Chart(ctx, {
              type: 'line',
              data: { labels: [], datasets: [{ label: 'best max-avg', data: [], tension: 0.15 }] },
              options: { animation: false, scales: { x: { title: {display:true, text:'sec'} }, y: { title: {display:true, text:'max-avg'} } } }
            });

            function refresh(){
              fetch("{{url_for('task_api', task_id=task_id)}}").then(r => r.json()).then(d => {
                document.getElementById('meta').innerHTML =
                  "Status: <b>"+d.status+"</b> — " +
                  "Algo: <code>"+d.algo+"</code>, Init: <code>"+d.init+"</code>, TL: <code>"+d.time_limit_s+"</code>s, Seed: <code>"+d.seed+"</code><br/>" +
                  "Istanza: <code>"+d.instance_dir+"</code>";

                if (d.trace && d.trace.length){
                  chart.data.labels = d.trace.map(p => p.t.toFixed(1));
                  chart.data.datasets[0].data = d.trace.map(p => p.max_avg_completion);
                  chart.update();
                }
                if (d.status === "done"){
                  document.getElementById('result').textContent = JSON.stringify(d.final, null, 2);
                } else if (d.status === "error"){
                  document.getElementById('result').textContent = "ERROR: " + (d.error || "");
                }
                if (d.status === "running" || d.status === "queued"){
                  setTimeout(refresh, 1200);
                }
              }).catch(e => {
                document.getElementById('result').textContent = "Errore fetch: " + e;
                setTimeout(refresh, 2000);
              });
            }
            refresh();
          </script>
        </body>
        </html>
        """
        return render_template_string(html, task_id=task_id)

    @app.get("/api/task/<task_id>")
    def task_api(task_id: str):
        with TASKS_LOCK:
            t = TASKS.get(task_id)
            if not t:
                return jsonify({"error": "task not found"}), 404
            # return lightweight payload
            return jsonify({
                "task_id": t["task_id"],
                "run_id": t["run_id"],
                "status": t["status"],
                "instance_dir": t["instance_dir"],
                "algo": t["algo"],
                "init": t["init"],
                "time_limit_s": t["time_limit_s"],
                "seed": t["seed"],
                "trace": [{"t": p["t"], "max_avg_completion": p["max_avg_completion"]} for p in t.get("trace", [])],
                "final": t.get("result"),
                "error": t.get("error"),
            })

    @app.get("/run/<run_id>")
    def view_run(run_id: str):
        jf = DEFAULT_RESULTS_DIR / f"run_{run_id}.json"
        if not jf.exists():
            return f"Run {run_id} not found.", 404
        data = json.loads(jf.read_text(encoding="utf-8"))

        html = """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>Run {{run_id}}</title>
          <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
          <style>
            body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
            .card { border: 1px solid #ddd; border-radius: 14px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 14px; }
            .muted { color: #666; font-size: 13px; }
            a { color: #0b63ce; text-decoration: none; }
            a:hover { text-decoration: underline; }
            code { background:#f6f6f6; padding:2px 6px; border-radius:8px; }
          </style>
        </head>
        <body>
          <h1>Run {{run_id}}</h1>
          <p><a href="{{url_for('index')}}">← Home</a> · <a href="{{url_for('compare')}}">Compare</a></p>

          <div class="card muted">
            <div><b>Istanza:</b> <code>{{data.instance_dir}}</code></div>
            <div><b>Algo:</b> <code>{{data.algo}}</code> · <b>Init:</b> <code>{{data.init}}</code> · <b>TL:</b> <code>{{data.time_limit_s}}</code>s · <b>Seed:</b> <code>{{data.seed}}</code></div>
            <div><b>Best max-avg:</b> {{'%.4f'|format(data.best_max_avg_completion)}} · <b>Best sum-avg:</b> {{'%.4f'|format(data.best_sum_avg_completion)}} · <b>Feasible:</b> {{data.feasible}}</div>
          </div>

          <div class="card">
            <h2>Curva best-so-far</h2>
            <canvas id="chart" height="110"></canvas>
          </div>

          <div class="card">
            <h2>JSON</h2>
            <pre class="muted">{{payload}}</pre>
          </div>

          <script>
            const trace = {{trace|safe}};
            const ctx = document.getElementById('chart');
            new Chart(ctx, {
              type: 'line',
              data: { labels: trace.map(p => p.t.toFixed(1)),
                      datasets: [{ label: 'best max-avg', data: trace.map(p => p.max_avg_completion), tension: 0.15 }] },
              options: { animation: false, scales: { x: { title: {display:true, text:'sec'} }, y: { title: {display:true, text:'max-avg'} } } }
            });
          </script>
        </body>
        </html>
        """
        trace = data.get("trace", [])
        return render_template_string(html, run_id=run_id, data=data, trace=json.dumps(trace), payload=json.dumps(data, indent=2))

    @app.get("/compare")
    def compare():
        # Load all saved dashboard runs
        runs = []
        for jf in sorted(DEFAULT_RESULTS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                runs.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:
                continue

        # Build simple pivot: best median per (algo, init, time_limit_s) across seeds/instances
        # (This is intentionally simple for demo; for thesis you can export CSVs and do analysis in notebook.)
        key_runs = {}
        for r in runs:
            k = (r.get("algo"), r.get("init"), int(r.get("time_limit_s", 0)))
            key_runs.setdefault(k, []).append(r.get("best_max_avg_completion", None))

        summary = []
        for (algo, init, tl), vals in key_runs.items():
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            vals_sorted = sorted(vals)
            median = vals_sorted[len(vals_sorted)//2]
            mean = sum(vals_sorted)/len(vals_sorted)
            summary.append({"algo": algo, "init": init, "time_limit_s": tl, "runs": len(vals_sorted), "median_best_maxavg": median, "mean_best_maxavg": mean})
        summary.sort(key=lambda x: (x["time_limit_s"], x["median_best_maxavg"]))

        html = """
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>Compare</title>
          <style>
            body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
            .card { border: 1px solid #ddd; border-radius: 14px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 14px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border-bottom: 1px solid #eee; padding: 8px 10px; font-size: 13px; text-align: left; }
            .muted { color: #666; font-size: 13px; }
            a { color: #0b63ce; text-decoration: none; }
            a:hover { text-decoration: underline; }
            code { background:#f6f6f6; padding:2px 6px; border-radius:8px; }
          </style>
        </head>
        <body>
          <h1>Compare</h1>
          <p><a href="{{url_for('index')}}">← Home</a></p>

          <div class="card">
            <h2>Classifica (mediana) per combinazione</h2>
            <p class="muted">Aggregazione sulle run salvate dalla dashboard (utile per demo). Per l’analisi “ufficiale” usa il batch runner e summary.csv.</p>
            {% if summary %}
              <table>
                <thead><tr>
                  <th>TL (s)</th><th>Algo</th><th>Init</th><th>#run</th><th>Median best max-avg</th><th>Mean best max-avg</th>
                </tr></thead>
                <tbody>
                  {% for s in summary %}
                    <tr>
                      <td>{{s.time_limit_s}}</td>
                      <td><code>{{s.algo}}</code></td>
                      <td><code>{{s.init}}</code></td>
                      <td>{{s.runs}}</td>
                      <td>{{"%.4f"|format(s.median_best_maxavg)}}</td>
                      <td>{{"%.4f"|format(s.mean_best_maxavg)}}</td>
                    </tr>
                  {% endfor %}
                </tbody>
              </table>
            {% else %}
              <p class="muted">Nessuna run salvata ancora. Esegui qualche run dalla Home.</p>
            {% endif %}
          </div>

          <div class="card">
            <h2>Run salvate</h2>
            {% if runs %}
              <table>
                <thead><tr>
                  <th>Quando</th><th>Algo</th><th>Init</th><th>TL</th><th>Best max-avg</th><th>Istanza</th><th></th>
                </tr></thead>
                <tbody>
                  {% for r in runs[:60] %}
                    <tr>
                      <td>{{r.ended_at}}</td>
                      <td><code>{{r.algo}}</code></td>
                      <td><code>{{r.init}}</code></td>
                      <td>{{r.time_limit_s}}</td>
                      <td>{{"%.4f"|format(r.best_max_avg_completion)}}</td>
                      <td class="muted">{{r.instance_dir}}</td>
                      <td><a href="{{url_for('view_run', run_id=r.run_id)}}">Apri</a></td>
                    </tr>
                  {% endfor %}
                </tbody>
              </table>
            {% else %}
              <p class="muted">Nessuna run.</p>
            {% endif %}
          </div>
        </body>
        </html>
        """
        return render_template_string(html, runs=runs, summary=summary)

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port)
