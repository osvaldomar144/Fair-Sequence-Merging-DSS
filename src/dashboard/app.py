from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, List

from flask import Flask, request, redirect, url_for, render_template_string, jsonify

from src.fairseq.runner import run_instance_trace

APP_TITLE = "Fair Sequence – Dashboard"
DEFAULT_RESULTS_DIR = Path("results/dashboard_runs")

# In-memory task store (fine for thesis/demo). For persistence, write task state to disk.
TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()


def _atomic_write_json(path: Path, payload: Dict[str, Any]):
    """Write JSON atomically (avoid partial reads during live update)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic rename on mac/linux


def _append_ndjson(path: Path, payload: Dict[str, Any]):
    """Append one JSON object per line (NDJSON)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


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

        def key(x: Path):
            return int(x.name) if x.name.isdigit() else x.name

        return [d.name for d in sorted(inst_dirs, key=key)]

    def safe_path(p: str) -> Path:
        return Path(p).expanduser().resolve()

    BOOTSTRAP_HEAD = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { background: #f7f7fb; }
      .card { border: 0; border-radius: 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .small-muted { color: #6c757d; font-size: 0.9rem; }
      .chip { display:inline-block; padding: .2rem .55rem; border-radius: 999px; font-size: .85rem; background: #eef1ff; }
      .table thead th { color: #6c757d; font-weight: 600; }
      .btn-dark { border-radius: 12px; }
      .form-control, .form-select { border-radius: 12px; }
      .shadow-soft { box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
    </style>
    """

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

        # Recent runs (from disk) — keep old behavior
        recent = []
        for jf in sorted(DEFAULT_RESULTS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:15]:
            # skip folders run_<id>/
            if jf.is_dir():
                continue
            try:
                recent.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:
                continue

        html = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>{{{{title}}}}</title>
          {BOOTSTRAP_HEAD}
        </head>
        <body>
          <div class="container py-4">
            <div class="d-flex align-items-end justify-content-between mb-3">
              <div>
                <h1 class="mb-1">{{{{title}}}}</h1>
                <div class="small-muted">Lancia run controllate e visualizza curve best-so-far (live).</div>
              </div>
              <a class="btn btn-outline-secondary" href="{{{{url_for('compare')}}}}">Compare</a>
            </div>

            <div class="row g-3">
              <div class="col-lg-6">
                <div class="card p-3">
                  <h5 class="mb-2">Nuova esecuzione</h5>
                  <form method="post" action="{{{{url_for('start_run')}}}}">
                    <div class="row g-2">
                      <div class="col-12">
                        <label class="form-label small-muted mb-1">Data root</label>
                        <input class="form-control" name="data_root" value="{{{{data_root}}}}" />
                      </div>

                      <div class="col-md-6">
                        <label class="form-label small-muted mb-1">Classe</label>
                        <select class="form-select" name="class" onchange="this.form.submit()">
                          {{% for c in classes %}}
                            <option value="{{{{c}}}}" {{% if c==cls %}}selected{{% endif %}}>{{{{c}}}}</option>
                          {{% endfor %}}
                        </select>
                      </div>

                      <div class="col-md-6">
                        <label class="form-label small-muted mb-1">Istanza</label>
                        <select class="form-select" name="instance">
                          {{% for i in instances %}}
                            <option value="{{{{i}}}}" {{% if i==inst %}}selected{{% endif %}}>{{{{i}}}}</option>
                          {{% endfor %}}
                        </select>
                      </div>

                      <div class="col-md-4">
                        <label class="form-label small-muted mb-1">Algoritmo</label>
                        <select class="form-select" name="algo">
                          {{% for a in ['ls','sa','tabu','ga'] %}}
                            <option value="{{{{a}}}}">{{{{a}}}}</option>
                          {{% endfor %}}
                        </select>
                      </div>

                      <div class="col-md-4">
                        <label class="form-label small-muted mb-1">Init</label>
                        <select class="form-select" name="init_method">
                          {{% for m in ['round_robin','random','greedy_fair'] %}}
                            <option value="{{{{m}}}}">{{{{m}}}}</option>
                          {{% endfor %}}
                        </select>
                      </div>

                      <div class="col-md-4">
                        <label class="form-label small-muted mb-1">Time limit (s)</label>
                        <input class="form-control" name="time_limit_s" type="number" value="180" min="1" step="1"/>
                      </div>

                      <div class="col-md-4">
                        <label class="form-label small-muted mb-1">Seed</label>
                        <input class="form-control" name="seed" type="number" value="1" min="0" step="1"/>
                      </div>

                      <div class="col-md-4">
                        <label class="form-label small-muted mb-1">Log every (s)</label>
                        <input class="form-control" name="log_every_s" type="number" value="1" min="0.2" step="0.2"/>
                      </div>

                      <div class="col-md-4 d-grid align-items-end">
                        <label class="form-label small-muted mb-1">&nbsp;</label>
                        <button class="btn btn-dark" type="submit">Avvia</button>
                      </div>
                    </div>
                  </form>

                  <div class="mt-3 small-muted">
                    Suggerimento: per i test finali usa <span class="chip">180 / 300 / 600</span> secondi e più seed.
                  </div>
                </div>
              </div>

              <div class="col-lg-6">
                <div class="card p-3">
                  <div class="d-flex justify-content-between align-items-center mb-2">
                    <h5 class="mb-0">Run recenti</h5>
                    <span class="small-muted">da file JSON</span>
                  </div>

                  {{% if recent %}}
                    <div class="table-responsive">
                      <table class="table table-sm align-middle mb-0">
                        <thead>
                          <tr>
                            <th>Quando</th><th>Algo</th><th>Init</th><th>TL</th><th>Best max-avg</th><th></th>
                          </tr>
                        </thead>
                        <tbody>
                          {{% for r in recent %}}
                            <tr>
                              <td class="small-muted">{{{{r.get('ended_at','')}}}}</td>
                              <td><span class="chip mono">{{{{r.get('algo','')}}}}</span></td>
                              <td><span class="chip mono">{{{{r.get('init','')}}}}</span></td>
                              <td class="mono">{{{{r.get('time_limit_s','')}}}}s</td>
                              <td class="mono">{{{{"%.4f"|format(r.get('best_max_avg_completion',0.0))}}}}</td>
                              <td><a href="{{{{url_for('view_run', run_id=r.get('run_id'))}}}}">Apri</a></td>
                            </tr>
                          {{% endfor %}}
                        </tbody>
                      </table>
                    </div>
                  {{% else %}}
                    <div class="small-muted">Nessuna run salvata ancora.</div>
                  {{% endif %}}
                </div>
              </div>
            </div>

            <div class="mt-3 small-muted">
              Live: mentre una run è in corso, vengono aggiornati <span class="mono">progress.json</span> e <span class="mono">trace.ndjson</span> in
              <span class="mono">results/dashboard_runs/run_&lt;run_id&gt;/</span>.
            </div>

          </div>
        </body>
        </html>
        """
        return render_template_string(
            html,
            title=APP_TITLE,
            data_root=data_root,
            classes=classes,
            cls=cls,
            instances=instances,
            inst=inst,
            recent=recent,
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

            # live-run folder
            run_dir = DEFAULT_RESULTS_DIR / f"run_{run_id}"
            progress_path = run_dir / "progress.json"
            trace_path = run_dir / "trace.ndjson"

            # initial progress
            _atomic_write_json(progress_path, {
                "run_id": run_id,
                "status": "running",
                "created_at": TASKS[task_id]["created_at"],
                "ended_at": "",
                "instance_dir": str(instance_dir),
                "algo": algo,
                "init": init_method,
                "time_limit_s": time_limit_s,
                "seed": seed,
                "log_every_s": log_every_s,
                "n_points": 0,
                "last_point": None,
                "best": None,
            })

            try:
                def live_recorder(elapsed: float, max_avg: float, sum_avg: float, obj: float):
                    p = {
                        "t": float(elapsed),
                        "max_avg_completion": float(max_avg),
                        "sum_avg_completion": float(sum_avg),
                        "obj": float(obj),
                    }
                    # 1) append to NDJSON
                    _append_ndjson(trace_path, p)

                    # 2) keep last ~2000 points in memory for live chart
                    with TASKS_LOCK:
                        tr = TASKS[task_id].get("trace", [])
                        tr.append(p)
                        if len(tr) > 2000:
                            tr[:] = tr[-2000:]
                        TASKS[task_id]["trace"] = tr

                    # 3) atomic progress update
                    _atomic_write_json(progress_path, {
                        "run_id": run_id,
                        "status": "running",
                        "created_at": TASKS[task_id]["created_at"],
                        "ended_at": "",
                        "instance_dir": str(instance_dir),
                        "algo": algo,
                        "init": init_method,
                        "time_limit_s": time_limit_s,
                        "seed": seed,
                        "log_every_s": log_every_s,
                        "n_points": len(TASKS[task_id].get("trace", [])),
                        "last_point": p,
                        "best": {
                            "best_obj": float(obj),
                            "best_max_avg_completion": float(max_avg),
                            "best_sum_avg_completion": float(sum_avg),
                        }
                    })

                # IMPORTANT: requires runner.py updated with external_recorder support
                res, trace = run_instance_trace(
                    Path(instance_dir),
                    algo=algo,
                    time_limit_s=time_limit_s,
                    seed=seed,
                    init_method=init_method,
                    log_every_s=log_every_s,
                    verbose=False,
                    external_recorder=live_recorder,  # <-- new
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

                # persist "single JSON" for recent runs list
                out = DEFAULT_RESULTS_DIR / f"run_{run_id}.json"
                out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                # final progress
                _atomic_write_json(progress_path, {
                    "run_id": run_id,
                    "status": "done",
                    "created_at": TASKS[task_id]["created_at"],
                    "ended_at": ended,
                    "instance_dir": str(instance_dir),
                    "algo": algo,
                    "init": init_method,
                    "time_limit_s": time_limit_s,
                    "seed": seed,
                    "log_every_s": log_every_s,
                    "n_points": len(trace),
                    "last_point": trace[-1] if trace else None,
                    "final": {
                        "best_obj": res.best_obj,
                        "best_max_avg_completion": res.best_max_avg,
                        "best_sum_avg_completion": res.best_sum_avg,
                        "feasible": res.feasible,
                        "violation": res.violation,
                        "n_iters": res.n_iters,
                        "elapsed_s": res.elapsed_s,
                    }
                })

                with TASKS_LOCK:
                    TASKS[task_id]["status"] = "done"
                    TASKS[task_id]["ended_at"] = ended
                    TASKS[task_id]["trace"] = trace  # full trace at end
                    TASKS[task_id]["result"] = asdict(res)

            except Exception as e:
                with TASKS_LOCK:
                    TASKS[task_id]["status"] = "error"
                    TASKS[task_id]["error"] = repr(e)
                try:
                    _atomic_write_json(progress_path, {
                        "run_id": run_id,
                        "status": "error",
                        "error": repr(e),
                    })
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()
        return redirect(url_for("task_page", task_id=task_id))

    @app.get("/task/<task_id>")
    def task_page(task_id: str):
        html = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>Run {{{{task_id}}}}</title>
          <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
          {BOOTSTRAP_HEAD}
        </head>
        <body>
          <div class="container py-4">
            <div class="d-flex justify-content-between align-items-center mb-3">
              <div>
                <h1 class="mb-1">Run in esecuzione</h1>
                <div class="small-muted">Aggiornamento live via API (polling).</div>
              </div>
              <div class="d-flex gap-2">
                <a class="btn btn-outline-secondary" href="{{{{url_for('index')}}}}">Home</a>
                <a class="btn btn-outline-secondary" href="{{{{url_for('compare')}}}}">Compare</a>
              </div>
            </div>

            <div class="row g-3">
              <div class="col-lg-5">
                <div class="card p-3">
                  <div id="meta" class="small-muted">Caricamento…</div>
                  <div class="mt-2">
                    <span id="badge" class="badge text-bg-secondary">...</span>
                  </div>
                </div>

                <div class="card p-3 mt-3">
                  <h6 class="mb-2">Esito</h6>
                  <pre id="result" class="small-muted mb-0" style="white-space: pre-wrap;">In attesa…</pre>
                </div>
              </div>

              <div class="col-lg-7">
                <div class="card p-3">
                  <div class="d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">Best max-avg completion vs tempo</h5>
                    <span class="small-muted">best-so-far</span>
                  </div>
                  <div class="small-muted mt-1">La curva scende quando viene trovata una soluzione migliore.</div>
                  <div class="mt-2">
                    <canvas id="chart" height="140"></canvas>
                  </div>
                </div>
              </div>
            </div>

          </div>

          <script>
            const taskId = "{{{{task_id}}}}";
            const ctx = document.getElementById('chart');
            const chart = new Chart(ctx, {{
              type: 'line',
              data: {{
                labels: [],
                datasets: [{{
                  label: 'best max-avg',
                  data: [],
                  tension: 0.15,
                  pointRadius: 0
                }}]
              }},
              options: {{
                animation: false,
                responsive: true,
                scales: {{
                  x: {{ title: {{display:true, text:'sec'}} }},
                  y: {{ title: {{display:true, text:'max-avg'}} }}
                }}
              }}
            }});

            function setBadge(status) {{
              const el = document.getElementById("badge");
              el.className = "badge";
              if (status === "done") el.classList.add("text-bg-success");
              else if (status === "running") el.classList.add("text-bg-primary");
              else if (status === "queued") el.classList.add("text-bg-secondary");
              else el.classList.add("text-bg-danger");
              el.textContent = status;
            }}

            function refresh(){{
              fetch("{{{{url_for('task_api', task_id=task_id)}}}}")
                .then(r => r.json())
                .then(d => {{
                  setBadge(d.status);

                  document.getElementById('meta').innerHTML =
                    "Status: <b>"+d.status+"</b><br/>" +
                    "Algo: <span class='chip mono'>"+d.algo+"</span> " +
                    "Init: <span class='chip mono'>"+d.init+"</span> " +
                    "TL: <span class='chip mono'>"+d.time_limit_s+"</span>s " +
                    "Seed: <span class='chip mono'>"+d.seed+"</span><br/>" +
                    "Istanza: <span class='mono'>"+d.instance_dir+"</span>";

                  if (d.trace && d.trace.length){{
                    chart.data.labels = d.trace.map(p => Number(p.t).toFixed(1));
                    chart.data.datasets[0].data = d.trace.map(p => p.max_avg_completion);
                    chart.update();
                  }}

                  if (d.status === "done"){{
                    document.getElementById('result').textContent = JSON.stringify(d.final, null, 2);
                  }} else if (d.status === "error"){{
                    document.getElementById('result').textContent = "ERROR: " + (d.error || "");
                  }}

                  if (d.status === "running" || d.status === "queued"){{
                    setTimeout(refresh, 1200);
                  }}
                }})
                .catch(e => {{
                  document.getElementById('result').textContent = "Errore fetch: " + e;
                  setTimeout(refresh, 2000);
                }});
            }}

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

            # return lightweight payload for live
            trace = t.get("trace", [])
            lite_trace = [{"t": p["t"], "max_avg_completion": p["max_avg_completion"]} for p in trace]

            return jsonify({
                "task_id": t["task_id"],
                "run_id": t["run_id"],
                "status": t["status"],
                "instance_dir": t["instance_dir"],
                "algo": t["algo"],
                "init": t["init"],
                "time_limit_s": t["time_limit_s"],
                "seed": t["seed"],
                "trace": lite_trace,
                "final": t.get("result"),
                "error": t.get("error"),
            })

    @app.get("/run/<run_id>")
    def view_run(run_id: str):
        jf = DEFAULT_RESULTS_DIR / f"run_{run_id}.json"
        if not jf.exists():
            return f"Run {run_id} not found.", 404
        data = json.loads(jf.read_text(encoding="utf-8"))

        html = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>Run {{{{run_id}}}}</title>
          <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
          {BOOTSTRAP_HEAD}
        </head>
        <body>
          <div class="container py-4">
            <div class="d-flex justify-content-between align-items-center mb-3">
              <div>
                <h1 class="mb-1">Run {{{{run_id}}}}</h1>
                <div class="small-muted">Dettaglio run salvata (file JSON completo).</div>
              </div>
              <div class="d-flex gap-2">
                <a class="btn btn-outline-secondary" href="{{{{url_for('index')}}}}">Home</a>
                <a class="btn btn-outline-secondary" href="{{{{url_for('compare')}}}}">Compare</a>
              </div>
            </div>

            <div class="card p-3 mb-3">
              <div class="row g-2 small-muted">
                <div class="col-12"><b>Istanza:</b> <span class="mono">{{{{data.instance_dir}}}}</span></div>
                <div class="col-12">
                  <b>Algo:</b> <span class="chip mono">{{{{data.algo}}}}</span>
                  <b class="ms-2">Init:</b> <span class="chip mono">{{{{data.init}}}}</span>
                  <b class="ms-2">TL:</b> <span class="chip mono">{{{{data.time_limit_s}}}}</span>s
                  <b class="ms-2">Seed:</b> <span class="chip mono">{{{{data.seed}}}}</span>
                </div>
                <div class="col-12">
                  <b>Best max-avg:</b> <span class="mono">{{{{'%.4f'|format(data.best_max_avg_completion)}}}}</span>
                  <b class="ms-2">Best sum-avg:</b> <span class="mono">{{{{'%.4f'|format(data.best_sum_avg_completion)}}}}</span>
                  <b class="ms-2">Feasible:</b> <span class="mono">{{{{data.feasible}}}}</span>
                  <b class="ms-2">Violation:</b> <span class="mono">{{{{data.violation}}}}</span>
                </div>
              </div>
            </div>

            <div class="card p-3 mb-3">
              <div class="d-flex justify-content-between align-items-center">
                <h5 class="mb-0">Curva best-so-far</h5>
                <span class="small-muted">max-avg</span>
              </div>
              <div class="mt-2">
                <canvas id="chart" height="140"></canvas>
              </div>
            </div>

            <div class="card p-3">
              <h6 class="mb-2">JSON</h6>
              <pre class="small-muted mb-0" style="white-space: pre-wrap;">{{{{payload}}}}</pre>
            </div>
          </div>

          <script>
            const trace = {{{{trace|safe}}}};
            const ctx = document.getElementById('chart');
            new Chart(ctx, {{
              type: 'line',
              data: {{
                labels: trace.map(p => Number(p.t).toFixed(1)),
                datasets: [{{
                  label: 'best max-avg',
                  data: trace.map(p => p.max_avg_completion),
                  tension: 0.15,
                  pointRadius: 0
                }}]
              }},
              options: {{
                animation: false,
                responsive: true,
                scales: {{
                  x: {{ title: {{display:true, text:'sec'}} }},
                  y: {{ title: {{display:true, text:'max-avg'}} }}
                }}
              }}
            }});
          </script>
        </body>
        </html>
        """
        trace = data.get("trace", [])
        return render_template_string(
            html,
            run_id=run_id,
            data=data,
            trace=json.dumps(trace),
            payload=json.dumps(data, indent=2),
        )

    @app.get("/compare")
    def compare():
        # Load all saved dashboard runs
        runs = []
        for jf in sorted(DEFAULT_RESULTS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if jf.is_dir():
                continue
            try:
                runs.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:
                continue

        # Simple pivot: best median per (algo, init, time_limit_s) across saved runs
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
            median = vals_sorted[len(vals_sorted) // 2]
            mean = sum(vals_sorted) / len(vals_sorted)
            summary.append({
                "algo": algo,
                "init": init,
                "time_limit_s": tl,
                "runs": len(vals_sorted),
                "median_best_maxavg": median,
                "mean_best_maxavg": mean
            })
        summary.sort(key=lambda x: (x["time_limit_s"], x["median_best_maxavg"]))

        html = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>Compare</title>
          {BOOTSTRAP_HEAD}
        </head>
        <body>
          <div class="container py-4">
            <div class="d-flex justify-content-between align-items-center mb-3">
              <div>
                <h1 class="mb-1">Compare</h1>
                <div class="small-muted">Aggregazione sulle run salvate dalla dashboard (utile per demo).</div>
              </div>
              <a class="btn btn-outline-secondary" href="{{{{url_for('index')}}}}">Home</a>
            </div>

            <div class="card p-3 mb-3">
              <h5 class="mb-1">Classifica (mediana) per combinazione</h5>
              <div class="small-muted mb-2">Per l’analisi “ufficiale” usa i batch runner e i CSV.</div>

              {{% if summary %}}
                <div class="table-responsive">
                  <table class="table table-sm align-middle mb-0">
                    <thead>
                      <tr>
                        <th>TL (s)</th><th>Algo</th><th>Init</th><th>#run</th><th>Median best max-avg</th><th>Mean best max-avg</th>
                      </tr>
                    </thead>
                    <tbody>
                      {{% for s in summary %}}
                        <tr>
                          <td class="mono">{{{{s.time_limit_s}}}}</td>
                          <td><span class="chip mono">{{{{s.algo}}}}</span></td>
                          <td><span class="chip mono">{{{{s.init}}}}</span></td>
                          <td class="mono">{{{{s.runs}}}}</td>
                          <td class="mono">{{{{"%.4f"|format(s.median_best_maxavg)}}}}</td>
                          <td class="mono">{{{{"%.4f"|format(s.mean_best_maxavg)}}}}</td>
                        </tr>
                      {{% endfor %}}
                    </tbody>
                  </table>
                </div>
              {{% else %}}
                <div class="small-muted">Nessuna run salvata ancora. Esegui qualche run dalla Home.</div>
              {{% endif %}}
            </div>

            <div class="card p-3">
              <h5 class="mb-2">Run salvate</h5>
              {{% if runs %}}
                <div class="table-responsive">
                  <table class="table table-sm align-middle mb-0">
                    <thead>
                      <tr>
                        <th>Quando</th><th>Algo</th><th>Init</th><th>TL</th><th>Best max-avg</th><th>Istanza</th><th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {{% for r in runs[:80] %}}
                        <tr>
                          <td class="small-muted">{{{{r.ended_at}}}}</td>
                          <td><span class="chip mono">{{{{r.algo}}}}</span></td>
                          <td><span class="chip mono">{{{{r.init}}}}</span></td>
                          <td class="mono">{{{{r.time_limit_s}}}}</td>
                          <td class="mono">{{{{"%.4f"|format(r.best_max_avg_completion)}}}}</td>
                          <td class="small-muted">{{{{r.instance_dir}}}}</td>
                          <td><a href="{{{{url_for('view_run', run_id=r.run_id)}}}}">Apri</a></td>
                        </tr>
                      {{% endfor %}}
                    </tbody>
                  </table>
                </div>
              {{% else %}}
                <div class="small-muted">Nessuna run.</div>
              {{% endif %}}
            </div>

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