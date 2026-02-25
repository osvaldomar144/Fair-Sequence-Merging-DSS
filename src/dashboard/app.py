from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from flask import Flask, request, redirect, url_for, render_template_string, jsonify

from src.fairseq.runner import run_instance_trace

APP_TITLE = "Fair Sequence – Dashboard"
DEFAULT_RESULTS_DIR = Path("results/dashboard_runs")

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
      .kpi { font-size: 1.05rem; font-weight: 700; }
    </style>
    """

    # --------------------------
    # NEW: helpers for grid page
    # --------------------------
    def _read_grid_csv(out_dir: Path):
        """
        Read progressive CSV produced by scripts/run_grid.py:
          raw_results.csv
        Return: (df or None, error or None, meta dict)
        """
        try:
            import pandas as pd  # local import (keeps app fast)
        except Exception as e:
            return None, f"pandas not available: {e!r}", {}

        out_dir = Path(out_dir)
        raw_path = out_dir / "raw_results.csv"
        meta = {
            "out_dir": str(out_dir),
            "raw_path": str(raw_path),
            "exists": raw_path.exists(),
            "mtime": raw_path.stat().st_mtime if raw_path.exists() else None,
            "size": raw_path.stat().st_size if raw_path.exists() else None,
        }
        if not raw_path.exists():
            return None, None, meta

        try:
            df = pd.read_csv(raw_path)
            # normalize column names if needed
            # expected columns:
            # class, instance, instance_name, algo, init, seed, time_limit_s,
            # feasible, violation, max_avg_completion, sum_avg_completion, iters, elapsed_s
            return df, None, meta
        except Exception as e:
            return None, f"failed reading CSV: {e!r}", meta

    def _compute_grid_views(df):
        """
        df -> (kpis dict, combo_df, best_by_instance_df)
        """
        import pandas as pd

        # safe casts
        if "feasible" in df.columns:
            df["feasible"] = df["feasible"].astype(bool)
        if "violation" in df.columns:
            df["violation"] = pd.to_numeric(df["violation"], errors="coerce").fillna(0.0)
        if "max_avg_completion" in df.columns:
            df["max_avg_completion"] = pd.to_numeric(df["max_avg_completion"], errors="coerce")
        if "sum_avg_completion" in df.columns:
            df["sum_avg_completion"] = pd.to_numeric(df["sum_avg_completion"], errors="coerce")

        kpis = {
            "n_rows": int(len(df)),
            "n_instances": int(df[["class", "instance_name"]].drop_duplicates().shape[0]) if "instance_name" in df.columns else int(df["instance"].nunique()) if "instance" in df.columns else 0,
            "n_families": int(df["class"].nunique()) if "class" in df.columns else 0,
            "feasible_rate": float(df["feasible"].mean()) if "feasible" in df.columns and len(df) else 0.0,
            "viol_mean": float(df["violation"].mean()) if "violation" in df.columns and len(df) else 0.0,
            "time_limits": sorted([int(x) for x in df["time_limit_s"].dropna().unique().tolist()]) if "time_limit_s" in df.columns else [],
            "seeds": sorted([int(x) for x in df["seed"].dropna().unique().tolist()]) if "seed" in df.columns else [],
            "algos": sorted(df["algo"].dropna().unique().tolist()) if "algo" in df.columns else [],
            "inits": sorted(df["init"].dropna().unique().tolist()) if "init" in df.columns else [],
        }

        # Summary by combo (algo, init, time_limit_s)
        if set(["algo", "init", "time_limit_s"]).issubset(df.columns):
            combo = df.groupby(["time_limit_s", "algo", "init"]).agg(
                runs=("max_avg_completion", "count"),
                feasible_rate=("feasible", "mean"),
                viol_mean=("violation", "mean"),
                viol_median=("violation", "median"),
                maxavg_mean=("max_avg_completion", "mean"),
                maxavg_median=("max_avg_completion", "median"),
                sumavg_mean=("sum_avg_completion", "mean"),
            ).reset_index()

            # sort: prioritize feasible, low violation, low maxavg
            combo = combo.sort_values(
                ["time_limit_s", "feasible_rate", "viol_median", "maxavg_median"],
                ascending=[True, False, True, True]
            )
        else:
            combo = pd.DataFrame()

        # Best by instance (per TL): pick best (algo, init) by:
        # 1) higher feasible_rate, 2) lower median_violation, 3) lower median maxavg
        if set(["class", "instance_name", "time_limit_s", "algo", "init"]).issubset(df.columns):
            best_rows = []
            for (fam, inst_name, tl), g in df.groupby(["class", "instance_name", "time_limit_s"]):
                g2 = g.groupby(["algo", "init"]).agg(
                    med=("max_avg_completion", "median"),
                    feas=("feasible", "mean"),
                    vmed=("violation", "median"),
                ).reset_index()

                g2 = g2.sort_values(["feas", "vmed", "med"], ascending=[False, True, True])
                top = g2.iloc[0]
                best_rows.append({
                    "class": fam,
                    "instance_name": inst_name,
                    "time_limit_s": int(tl),
                    "best_algo": top["algo"],
                    "best_init": top["init"],
                    "best_median_maxavg": float(top["med"]),
                    "feasible_rate": float(top["feas"]),
                    "median_violation": float(top["vmed"]),
                })
            best_by_instance = pd.DataFrame(best_rows).sort_values(["time_limit_s", "best_median_maxavg"])
        else:
            best_by_instance = pd.DataFrame()

        return kpis, combo, best_by_instance

    # --------------------------
    # Home
    # --------------------------
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

        recent = []
        for jf in sorted(DEFAULT_RESULTS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:15]:
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
              <div class="d-flex gap-2">
                <a class="btn btn-outline-secondary" href="{{{{url_for('grid_page')}}}}">Grid results</a>
                <a class="btn btn-outline-secondary" href="{{{{url_for('compare')}}}}">Compare</a>
              </div>
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
                    Per i test finali userò <span class="chip">180 / 300 / 600</span> secondi.
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
              Osvaldo Alexis Hidalgo Martinez <br>
              <strong>Matricola:</strong> 549484
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

    # --------------------------
    # NEW: Grid results page
    # --------------------------
    @app.get("/grid")
    def grid_page():
        out_dir = request.args.get("out_dir", "results/grid_now_5min")
        refresh_s = float(request.args.get("refresh_s", "10"))
        out_dir_p = safe_path(out_dir)

        df, err, meta = _read_grid_csv(out_dir_p)
        kpis = {}
        combo_rows = []
        best_rows = []

        if df is not None and err is None and len(df) > 0:
            kpis, combo_df, best_df = _compute_grid_views(df)
            combo_rows = combo_df.head(50).to_dict(orient="records") if len(combo_df) else []
            best_rows = best_df.head(80).to_dict(orient="records") if len(best_df) else []

        html = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>Grid results</title>
          {BOOTSTRAP_HEAD}
          <script>
            setTimeout(() => {{
              const url = new URL(window.location.href);
              const refreshS = Number(url.searchParams.get("refresh_s") || "{refresh_s}");
              if (refreshS > 0) window.location.reload();
            }}, {int(refresh_s*1000)});
          </script>
        </head>
        <body>
          <div class="container py-4">
            <div class="d-flex justify-content-between align-items-center mb-3">
              <div>
                <h1 class="mb-1">Grid results</h1>
                <div class="small-muted">Legge <span class="mono">raw_results.csv</span> in modo progressivo (auto-refresh).</div>
              </div>
              <div class="d-flex gap-2">
                <a class="btn btn-outline-secondary" href="{{{{url_for('index')}}}}">Home</a>
                <a class="btn btn-outline-secondary" href="{{{{url_for('compare')}}}}">Compare</a>
              </div>
            </div>

            <div class="card p-3 mb-3">
              <form method="get" action="{{{{url_for('grid_page')}}}}">
                <div class="row g-2 align-items-end">
                  <div class="col-md-7">
                    <label class="form-label small-muted mb-1">Out dir (run_grid.py)</label>
                    <input class="form-control mono" name="out_dir" value="{{{{out_dir}}}}" />
                    <div class="small-muted mt-1">Esempio: <span class="mono">results/grid_now_5min</span></div>
                  </div>
                  <div class="col-md-3">
                    <label class="form-label small-muted mb-1">Auto-refresh (sec)</label>
                    <input class="form-control mono" name="refresh_s" type="number" min="0" step="1" value="{{{{refresh_s}}}}" />
                    <div class="small-muted mt-1">0 = off</div>
                  </div>
                  <div class="col-md-2 d-grid">
                    <button class="btn btn-dark" type="submit">Apri</button>
                  </div>
                </div>
              </form>
            </div>

            <div class="row g-3">
              <div class="col-lg-4">
                <div class="card p-3">
                  <h5 class="mb-2">Stato</h5>
                  <div class="small-muted">out_dir: <span class="mono">{{{{meta.out_dir}}}}</span></div>
                  <div class="small-muted">raw_results.csv: <span class="mono">{{{{meta.raw_path}}}}</span></div>
                  <div class="small-muted">exists: <span class="mono">{{{{meta.exists}}}}</span></div>
                  <div class="small-muted">size: <span class="mono">{{{{meta.size}}}}</span></div>
                  <div class="small-muted">mtime: <span class="mono">{{{{meta.mtime_readable}}}}</span></div>

                  {{% if err %}}
                    <div class="alert alert-danger mt-2 mb-0"><b>Errore:</b> {{{{err}}}}</div>
                  {{% endif %}}
                  {{% if not meta.exists %}}
                    <div class="alert alert-warning mt-2 mb-0">File non trovato. Avvia run_grid.py o controlla l’out_dir.</div>
                  {{% endif %}}
                </div>

                <div class="card p-3 mt-3">
                  <h5 class="mb-2">KPI (parziali)</h5>
                  {{% if kpis %}}
                    <div class="row g-2">
                      <div class="col-6"><div class="small-muted">righe</div><div class="kpi mono">{{{{kpis.n_rows}}}}</div></div>
                      <div class="col-6"><div class="small-muted">istanze</div><div class="kpi mono">{{{{kpis.n_instances}}}}</div></div>
                      <div class="col-6"><div class="small-muted">famiglie</div><div class="kpi mono">{{{{kpis.n_families}}}}</div></div>
                      <div class="col-6"><div class="small-muted">feasible rate</div><div class="kpi mono">{{{{"%.2f"|format(100*kpis.feasible_rate)}}}}%</div></div>
                      <div class="col-12"><div class="small-muted">viol mean</div><div class="kpi mono">{{{{"%.3f"|format(kpis.viol_mean)}}}}</div></div>
                    </div>
                  {{% else %}}
                    <div class="small-muted">In attesa di righe nel CSV…</div>
                  {{% endif %}}
                </div>
              </div>

              <div class="col-lg-8">
                <div class="card p-3 mb-3">
                  <div class="d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">Ranking per combinazione (TL, algo, init)</h5>
                    <span class="small-muted">top 50</span>
                  </div>

                  {{% if combo_rows %}}
                    <div class="table-responsive mt-2">
                      <table class="table table-sm align-middle mb-0">
                        <thead>
                          <tr>
                            <th>TL</th><th>Algo</th><th>Init</th><th>#</th><th>Feas%</th><th>Viol med</th><th>MaxAvg med</th><th>MaxAvg mean</th>
                          </tr>
                        </thead>
                        <tbody>
                          {{% for r in combo_rows %}}
                            <tr>
                              <td class="mono">{{{{r.time_limit_s|int}}}}</td>
                              <td><span class="chip mono">{{{{r.algo}}}}</span></td>
                              <td><span class="chip mono">{{{{r.init}}}}</span></td>
                              <td class="mono">{{{{r.runs|int}}}}</td>
                              <td class="mono">{{{{"%.0f"|format(100*r.feasible_rate)}}}}%</td>
                              <td class="mono">{{{{"%.3f"|format(r.viol_median)}}}}</td>
                              <td class="mono">{{{{"%.4f"|format(r.maxavg_median)}}}}</td>
                              <td class="mono">{{{{"%.4f"|format(r.maxavg_mean)}}}}</td>
                            </tr>
                          {{% endfor %}}
                        </tbody>
                      </table>
                    </div>
                  {{% else %}}
                    <div class="small-muted mt-2">Nessun dato ancora.</div>
                  {{% endif %}}
                </div>

                <div class="card p-3">
                  <div class="d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">Best per istanza (per TL)</h5>
                    <span class="small-muted">top 80</span>
                  </div>

                  {{% if best_rows %}}
                    <div class="table-responsive mt-2">
                      <table class="table table-sm align-middle mb-0">
                        <thead>
                          <tr>
                            <th>Fam</th><th>Inst</th><th>TL</th><th>Best algo</th><th>Best init</th><th>Feas%</th><th>Viol med</th><th>MaxAvg med</th>
                          </tr>
                        </thead>
                        <tbody>
                          {{% for r in best_rows %}}
                            <tr>
                              <td class="mono">{{{{r["class"]}}}}</td>
                              <td class="mono">{{{{r["instance_name"]}}}}</td>
                              <td class="mono">{{{{r["time_limit_s"]}}}}</td>
                              <td><span class="chip mono">{{{{r["best_algo"]}}}}</span></td>
                              <td><span class="chip mono">{{{{r["best_init"]}}}}</span></td>
                              <td class="mono">{{{{"%.0f"|format(100*r["feasible_rate"])}}}}%</td>
                              <td class="mono">{{{{"%.3f"|format(r["median_violation"])}}}}</td>
                              <td class="mono">{{{{"%.4f"|format(r["best_median_maxavg"])}}}}</td>
                            </tr>
                          {{% endfor %}}
                        </tbody>
                      </table>
                    </div>
                  {{% else %}}
                    <div class="small-muted mt-2">Nessun dato ancora.</div>
                  {{% endif %}}
                </div>

              </div>
            </div>
          </div>
        </body>
        </html>
        """

        # render meta as an object-like dict for jinja
        meta_for_tpl = {
            "out_dir": meta.get("out_dir", ""),
            "raw_path": meta.get("raw_path", ""),
            "exists": bool(meta.get("exists", False)),
            "size": meta.get("size", ""),
            "mtime_readable": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta["mtime"])) if meta.get("mtime") else "",
        }

        return render_template_string(
            html,
            out_dir=out_dir,
            refresh_s=refresh_s,
            meta=meta_for_tpl,
            err=err,
            kpis=kpis,
            combo_rows=combo_rows,
            best_rows=best_rows,
        )

    # Optional: API endpoint (se ti serve in futuro per polling più “smart”)
    @app.get("/api/grid_status")
    def grid_status_api():
        out_dir = request.args.get("out_dir", "results/grid_now_5min")
        out_dir_p = safe_path(out_dir)
        df, err, meta = _read_grid_csv(out_dir_p)
        if df is None or err is not None or len(df) == 0:
            return jsonify({"ok": False, "error": err, "meta": meta, "n_rows": 0})
        kpis, combo_df, best_df = _compute_grid_views(df)
        return jsonify({
            "ok": True,
            "meta": meta,
            "kpis": kpis,
            "combo_top": combo_df.head(20).to_dict(orient="records") if len(combo_df) else [],
            "best_by_instance_top": best_df.head(20).to_dict(orient="records") if len(best_df) else [],
        })

    # --------------------------
    # Existing: start live run
    # --------------------------
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

            run_dir = DEFAULT_RESULTS_DIR / f"run_{run_id}"
            progress_path = run_dir / "progress.json"
            trace_path = run_dir / "trace.ndjson"

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

                    _append_ndjson(trace_path, p)

                    with TASKS_LOCK:
                        tr = TASKS[task_id].get("trace", [])
                        tr.append(p)
                        if len(tr) > 2000:
                            tr[:] = tr[-2000:]
                        TASKS[task_id]["trace"] = tr

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

                res, trace, diagnostics = run_instance_trace(
                    Path(instance_dir),
                    algo=algo,
                    time_limit_s=time_limit_s,
                    seed=seed,
                    init_method=init_method,
                    log_every_s=log_every_s,
                    verbose=False,
                    external_recorder=live_recorder,
                    return_diagnostics=True,
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
                    "diagnostics": diagnostics,
                }

                out = DEFAULT_RESULTS_DIR / f"run_{run_id}.json"
                out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
                    TASKS[task_id]["trace"] = trace
                    TASKS[task_id]["diagnostics"] = diagnostics
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

    # --------------------------
    # Existing: task page + api
    # --------------------------
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
                <a class="btn btn-outline-secondary" href="{{{{url_for('grid_page')}}}}">Grid results</a>
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
                "diagnostics": t.get("diagnostics"),
                "final": t.get("result"),
                "error": t.get("error"),
            })

    # --------------------------
    # Existing: view run
    # --------------------------
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
                <a class="btn btn-outline-secondary" href="{{{{url_for('grid_page')}}}}">Grid results</a>
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

    # --------------------------
    # Existing: compare
    # --------------------------
    @app.get("/compare")
    def compare():
        runs = []
        for jf in sorted(DEFAULT_RESULTS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if jf.is_dir():
                continue
            try:
                runs.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:
                continue

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
              <div class="d-flex gap-2">
                <a class="btn btn-outline-secondary" href="{{{{url_for('index')}}}}">Home</a>
                <a class="btn btn-outline-secondary" href="{{{{url_for('grid_page')}}}}">Grid results</a>
              </div>
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