from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from flask import Flask, request, redirect, url_for, jsonify, render_template

from src.fairseq.runner import run_instance_trace

APP_TITLE = "Fair Sequence – Dashboard"
DEFAULT_RESULTS_DIR = Path("results/dashboard_runs")

QUEUE_STATE_PATH = DEFAULT_RESULTS_DIR / "queue_state.json"
QUEUE_LOCK = threading.Lock()

TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()


# --------------------------
# Utils
# --------------------------
def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]):
    _atomic_write_text(path, json.dumps(payload, indent=2))


def _append_ndjson(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def safe_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


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


# --------------------------
# Queue persistence
# --------------------------
def _load_queue_state() -> Dict[str, Any]:
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not QUEUE_STATE_PATH.exists():
        return {
            "queue_id": uuid.uuid4().hex[:10],
            "status": "idle",
            "created_at": _now_str(),
            "started_at": "",
            "ended_at": "",
            "current_job_id": "",
            "jobs": [],
            "current_trace": [],
            "current_best": None,
            "last_error": None,
        }

    try:
        data = json.loads(QUEUE_STATE_PATH.read_text(encoding="utf-8"))
        data.setdefault("queue_id", uuid.uuid4().hex[:10])
        data.setdefault("status", "idle")
        data.setdefault("created_at", _now_str())
        data.setdefault("started_at", "")
        data.setdefault("ended_at", "")
        data.setdefault("current_job_id", "")
        data.setdefault("jobs", [])
        data.setdefault("current_trace", [])
        data.setdefault("current_best", None)
        data.setdefault("last_error", None)

        # se era running e hai riavviato Flask, lo rimettiamo idle
        if data.get("status") == "running":
            data["status"] = "idle"
            data["current_job_id"] = ""
        return data

    except Exception:
        return {
            "queue_id": uuid.uuid4().hex[:10],
            "status": "idle",
            "created_at": _now_str(),
            "started_at": "",
            "ended_at": "",
            "current_job_id": "",
            "jobs": [],
            "current_trace": [],
            "current_best": None,
            "last_error": "failed to read queue_state.json (reset)",
        }


def _save_queue_state(state: Dict[str, Any]):
    _atomic_write_text(QUEUE_STATE_PATH, json.dumps(state, indent=2))


def _queue_has_pending(st: Dict[str, Any]) -> bool:
    return any(j.get("status") == "queued" for j in st.get("jobs", []))


def _queue_counts(st: Dict[str, Any]) -> Dict[str, int]:
    jobs = st.get("jobs", [])
    return {
        "queued": sum(1 for j in jobs if j.get("status") == "queued"),
        "running": sum(1 for j in jobs if j.get("status") == "running"),
        "done": sum(1 for j in jobs if j.get("status") == "done"),
        "error": sum(1 for j in jobs if j.get("status") == "error"),
        "total": len(jobs),
    }


def _is_init_allowed(algo: str, init_method: str) -> Tuple[bool, Optional[str]]:
    if algo == "ga" and init_method == "round_robin":
        return False, "Il Genetic Algorithm non supporta l’inizializzazione round_robin. Seleziona random o greedy_fair."
    return True, None


# --------------------------
# Flask app
# --------------------------
def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Queue helpers ----------
    def queue_state() -> Dict[str, Any]:
        with QUEUE_LOCK:
            return _load_queue_state()

    def queue_add_job(payload: Dict[str, Any]) -> str:
        with QUEUE_LOCK:
            st = _load_queue_state()
            job_id = uuid.uuid4().hex[:10]
            job = {
                "job_id": job_id,
                "created_at": _now_str(),
                "started_at": "",
                "ended_at": "",
                "status": "queued",
                "error": None,
                "result": None,
                **payload,
            }
            st["jobs"].append(job)
            st["ended_at"] = ""
            st["last_error"] = None
            _save_queue_state(st)
            return job_id

    def queue_clear() -> bool:
        with QUEUE_LOCK:
            st = _load_queue_state()
            if st.get("status") == "running":
                return False
            st["jobs"] = []
            st["current_job_id"] = ""
            st["current_trace"] = []
            st["current_best"] = None
            st["status"] = "idle"
            st["started_at"] = ""
            st["ended_at"] = ""
            st["last_error"] = None
            _save_queue_state(st)
            return True

    def _queue_worker():
        with QUEUE_LOCK:
            st = _load_queue_state()
            if st.get("status") == "running":
                return
            st["status"] = "running"
            st["started_at"] = _now_str()
            st["ended_at"] = ""
            st["last_error"] = None
            st["current_job_id"] = ""
            st["current_trace"] = []
            st["current_best"] = None
            _save_queue_state(st)

        try:
            while True:
                with QUEUE_LOCK:
                    st = _load_queue_state()
                    job = next((j for j in st.get("jobs", []) if j.get("status") == "queued"), None)

                    if job is None:
                        st["status"] = "done" if len(st.get("jobs", [])) else "idle"
                        st["current_job_id"] = ""
                        st["ended_at"] = _now_str() if st["status"] != "idle" else ""
                        _save_queue_state(st)
                        return

                    job_id = job["job_id"]
                    st["current_job_id"] = job_id
                    st["current_trace"] = []
                    st["current_best"] = None

                    for jj in st["jobs"]:
                        if jj.get("job_id") == job_id:
                            jj["status"] = "running"
                            jj["started_at"] = _now_str()
                            jj["ended_at"] = ""
                            jj["error"] = None
                            jj["result"] = None
                            break

                    _save_queue_state(st)

                instance_dir = Path(job["instance_dir"])
                algo = job["algo"]
                init_method = job["init"]
                time_limit_s = int(job["time_limit_s"])
                seed = int(job["seed"])
                log_every_s = float(job.get("log_every_s", 1.0))

                ok, msg = _is_init_allowed(algo, init_method)
                if not ok:
                    with QUEUE_LOCK:
                        st3 = _load_queue_state()
                        st3["last_error"] = msg
                        for jj in st3.get("jobs", []):
                            if jj.get("job_id") == job_id:
                                jj["status"] = "error"
                                jj["ended_at"] = _now_str()
                                jj["error"] = msg
                                break
                        _save_queue_state(st3)
                    continue

                def live_recorder(elapsed: float, max_avg: float, sum_avg: float, obj: float):
                    p = {"t": float(elapsed), "max_avg_completion": float(max_avg), "sum_avg_completion": float(sum_avg), "obj": float(obj)}
                    with QUEUE_LOCK:
                        st2 = _load_queue_state()
                        if st2.get("current_job_id") != job_id:
                            return
                        tr = st2.get("current_trace", [])
                        tr.append(p)
                        if len(tr) > 2000:
                            tr = tr[-2000:]
                        st2["current_trace"] = tr
                        st2["current_best"] = {
                            "best_obj": float(obj),
                            "best_max_avg_completion": float(max_avg),
                            "best_sum_avg_completion": float(sum_avg),
                        }
                        _save_queue_state(st2)

                try:
                    res, trace, diagnostics = run_instance_trace(
                        instance_dir,
                        algo=algo,
                        time_limit_s=time_limit_s,
                        seed=seed,
                        init_method=init_method,
                        log_every_s=log_every_s,
                        verbose=False,
                        external_recorder=live_recorder,
                        return_diagnostics=True,
                    )

                    ended = _now_str()
                    run_id = f"q_{job_id}"
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
                        "queue_job_id": job_id,
                    }
                    (DEFAULT_RESULTS_DIR / f"run_{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

                    with QUEUE_LOCK:
                        st3 = _load_queue_state()
                        for jj in st3.get("jobs", []):
                            if jj.get("job_id") == job_id:
                                jj["status"] = "done"
                                jj["ended_at"] = ended
                                jj["result"] = {
                                    "best_obj": float(res.best_obj),
                                    "best_max_avg": float(res.best_max_avg),
                                    "best_sum_avg": float(res.best_sum_avg),
                                    "feasible": bool(res.feasible),
                                    "violation": float(res.violation),
                                    "n_iters": int(res.n_iters),
                                    "elapsed_s": float(res.elapsed_s),
                                }
                                break
                        _save_queue_state(st3)

                except Exception as e:
                    with QUEUE_LOCK:
                        st3 = _load_queue_state()
                        st3["last_error"] = repr(e)
                        for jj in st3.get("jobs", []):
                            if jj.get("job_id") == job_id:
                                jj["status"] = "error"
                                jj["ended_at"] = _now_str()
                                jj["error"] = repr(e)
                                break
                        _save_queue_state(st3)

        except Exception as e:
            with QUEUE_LOCK:
                st = _load_queue_state()
                st["status"] = "error"
                st["ended_at"] = _now_str()
                st["last_error"] = repr(e)
                _save_queue_state(st)

    # ---------- MENU + GLOBALS ----------
    @app.context_processor
    def inject_globals():
        st = queue_state()
        counts = _queue_counts(st)
        jobs_preview = list(reversed(st.get("jobs", [])))[:12]
        return dict(
            APP_TITLE=APP_TITLE,
            nav_items=[
                {"label": "Home", "endpoint": "index"},
                {"label": "Queue", "endpoint": "queue_page"},
                {"label": "Specific summary", "endpoint": "specific_summary_page"},
                {"label": "History", "endpoint": "history"},
            ],
            queue_status=st.get("status", "idle"),
            queue_counts=counts,
            queue_current_job_id=st.get("current_job_id", ""),
            queue_last_error=st.get("last_error"),
            queue_jobs_preview=jobs_preview,
        )

    # --------------------------
    # Home
    # --------------------------
    @app.get("/")
    def index():
        data_root = request.args.get("data_root", "data")
        data_root_p = safe_path(data_root)

        cls = request.args.get("class")
        classes = list_classes(data_root_p)
        if cls is None and classes:
            cls = classes[0]

        instances = list_instances(data_root_p, cls) if cls else []
        inst = request.args.get("instance")
        if inst is None and instances:
            inst = instances[0]

        msg = request.args.get("msg", "")

        # recent runs still computed (template may or may not show them)
        recent = []
        for jf in sorted(DEFAULT_RESULTS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:15]:
            if jf.is_dir():
                continue
            try:
                recent.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:
                continue

        return render_template(
            "index.html",
            title=APP_TITLE,
            data_root=data_root,
            classes=classes,
            cls=cls,
            instances=instances,
            inst=inst,
            recent=recent,
            msg=msg,
        )

    # --------------------------
    # Queue endpoints
    # --------------------------
    @app.post("/queue/add")
    def queue_add():
        data_root = request.form.get("data_root", "data")
        cls = request.form.get("class")
        inst = request.form.get("instance")
        algo = request.form.get("algo", "sa")
        init_method = request.form.get("init_method", "round_robin")
        time_limit_s = int(float(request.form.get("time_limit_s", "180")))
        seed = int(float(request.form.get("seed", "1")))
        log_every_s = float(request.form.get("log_every_s", "1"))

        ok, err = _is_init_allowed(algo, init_method)
        if not ok:
            return redirect(url_for("index", data_root=data_root, **{"class": cls, "instance": inst, "msg": err}))

        instance_dir = safe_path(data_root) / cls / inst
        queue_add_job({
            "instance_dir": str(instance_dir),
            "algo": algo,
            "init": init_method,
            "time_limit_s": time_limit_s,
            "seed": seed,
            "log_every_s": log_every_s,
        })
        return redirect(url_for("index", data_root=data_root, **{"class": cls, "instance": inst}))

    @app.post("/queue/clear")
    def queue_clear_route():
        queue_clear()
        return redirect(url_for("queue_page"))

    @app.post("/queue/start")
    def queue_start():
        st = queue_state()
        if st.get("status") != "running" and _queue_has_pending(st):
            threading.Thread(target=_queue_worker, daemon=True).start()
        return redirect(url_for("queue_page"))

    @app.get("/api/queue")
    def queue_api():
        st = queue_state()
        return jsonify({
            "queue_id": st.get("queue_id"),
            "status": st.get("status"),
            "created_at": st.get("created_at"),
            "started_at": st.get("started_at"),
            "ended_at": st.get("ended_at"),
            "current_job_id": st.get("current_job_id"),
            "current_best": st.get("current_best"),
            "current_trace": [{"t": p["t"], "max_avg_completion": p["max_avg_completion"]} for p in st.get("current_trace", [])],
            "jobs": st.get("jobs", []),
            "last_error": st.get("last_error"),
        })

    @app.get("/queue")
    def queue_page():
        return render_template("queue.html", title=APP_TITLE)

    # --------------------------
    # Specific summary (NEW)
    # --------------------------
    @app.get("/specific-summary")
    def specific_summary_page():
        out_dir = request.args.get("out_dir", "results/specific_summary_test")
        refresh_s = float(request.args.get("refresh_s", "10"))

        out_dir_p = safe_path(out_dir)
        json_path = out_dir_p / "specific_summary.json"

        payload = None
        error = None
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                error = f"Errore leggendo {json_path}: {e!r}"
        else:
            error = f"File non trovato: {json_path}"

        return render_template(
            "specific_summary.html",
            title=APP_TITLE,
            out_dir=out_dir,
            refresh_s=refresh_s,
            payload=payload,
            error=error,
        )

    # --------------------------
    # Old /grid kept as stub (optional)
    # --------------------------
    @app.get("/grid")
    def grid_page():
        out_dir = request.args.get("out_dir", "results/grid_now_5min")
        refresh_s = float(request.args.get("refresh_s", "10"))
        return render_template("grid_stub.html", title=APP_TITLE, out_dir=out_dir, refresh_s=refresh_s)

    # --------------------------
    # Live single run
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

        ok, err = _is_init_allowed(algo, init_method)
        if not ok:
            return redirect(url_for("index", data_root=data_root, **{"class": cls, "instance": inst, "msg": err}))

        instance_dir = safe_path(data_root) / cls / inst

        run_id = uuid.uuid4().hex[:10]
        task_id = uuid.uuid4().hex

        task = {
            "task_id": task_id,
            "run_id": run_id,
            "status": "queued",
            "created_at": _now_str(),
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
            trace_path = run_dir / "trace.ndjson"

            try:
                def live_recorder(elapsed: float, max_avg: float, sum_avg: float, obj: float):
                    p = {"t": float(elapsed), "max_avg_completion": float(max_avg), "sum_avg_completion": float(sum_avg), "obj": float(obj)}
                    _append_ndjson(trace_path, p)
                    with TASKS_LOCK:
                        tr = TASKS[task_id].get("trace", [])
                        tr.append(p)
                        if len(tr) > 2000:
                            tr[:] = tr[-2000:]
                        TASKS[task_id]["trace"] = tr

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

                ended = _now_str()
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
                (DEFAULT_RESULTS_DIR / f"run_{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

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

        threading.Thread(target=worker, daemon=True).start()
        return redirect(url_for("task_page", task_id=task_id))

    @app.get("/task/<task_id>")
    def task_page(task_id: str):
        return render_template("task.html", title=APP_TITLE, task_id=task_id)

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
                "final": t.get("result"),
                "error": t.get("error"),
            })

    # --------------------------
    # View run JSON
    # --------------------------
    @app.get("/run/<run_id>")
    def view_run(run_id: str):
        jf = DEFAULT_RESULTS_DIR / f"run_{run_id}.json"
        if not jf.exists():
            return f"Run {run_id} not found.", 404
        data = json.loads(jf.read_text(encoding="utf-8"))
        return render_template("run.html", title=APP_TITLE, run_id=run_id, data=data)

    # --------------------------
    # History (was /compare)
    # --------------------------
    @app.get("/history")
    def history():
        runs = []
        for jf in sorted(DEFAULT_RESULTS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if jf.is_dir():
                continue
            try:
                runs.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception:
                continue

        key_runs: Dict[Tuple[str, str, int], List[float]] = {}
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
        return render_template("compare.html", title=APP_TITLE, summary=summary, runs=runs[:80])

    # Backward-compatible route
    @app.get("/compare")
    def compare():
        return redirect(url_for("history"))

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port)