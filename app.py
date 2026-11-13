from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from analysis import (
    infer_columns, validate_interactions, build_graph, node_features,
    community_analysis, repeated_targeting, temporal_bursts, train_models,
    explain_user
)
from database import (
    init_database, save_dataset, load_latest_dataset, save_model_run,
    get_db_status
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["JSON_SORT_KEYS"] = False

STATE = {
    "df": None, "interactions": None, "G": None, "table": None, "X": None,
    "features": None, "communities": None, "repeated": None, "bursts": None,
    "models": None, "dataset_name": None, "is_demo": False, "dataset_id": None,
}

def json_safe(value):
    """Convert numpy/pandas values and timestamps to JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value

def _load_dataframe(df: pd.DataFrame, name: str, is_demo: bool = False, persist: bool = True):
    interactions = validate_interactions(df)
    G, nodes = build_graph(interactions)
    table, X, features = node_features(G, interactions, nodes)
    communities = community_analysis(G)
    repeated = repeated_targeting(interactions)
    bursts = temporal_bursts(interactions)
    dataset_id = None
    if persist:
        dataset_id = save_dataset(name, interactions, is_demo)
    STATE.update({
        "df": df, "interactions": interactions, "G": G, "table": table, "X": X,
        "features": features, "communities": communities, "repeated": repeated,
        "bursts": bursts, "models": None, "dataset_name": name,
        "is_demo": is_demo, "dataset_id": dataset_id,
    })
    return _summary()

def _summary():
    if STATE["G"] is None:
        return {"loaded": False}
    G, I, T = STATE["G"], STATE["interactions"], STATE["table"]
    return {
        "loaded": True,
        "dataset_id": STATE["dataset_id"],
        "dataset_name": STATE["dataset_name"],
        "is_demo": STATE["is_demo"],
        "rows": int(len(I)),
        "users": int(G.number_of_nodes()),
        "directed_edges": int(G.number_of_edges()),
        "communities": int(len(set(STATE["communities"].values()))),
        "labelled_interactions": int(I.is_abusive.notna().sum()),
        "abusive_interactions": int((I.is_abusive == 1).sum()),
        "labelled_users": int(T.label.notna().sum()),
        "repeated_targets": len(STATE["repeated"]),
        "temporal_bursts": len(STATE["bursts"]),
        "feature_names": STATE["features"],
    }

def _require_graph():
    if STATE["G"] is None:
        return jsonify({"error": "Load a dataset first."}), 409
    return None

@app.before_request
def ensure_schema():
                                                                             
    if not app.config.get("_db_checked"):
        init_database()
        app.config["_db_checked"] = True

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/health")
def health():
    try:
        db = get_db_status()
        return jsonify({"status": "ok", "service": "GNN Harassment Network Analysis", "database": db})
    except Exception as exc:
        return jsonify({"status": "degraded", "database": "unavailable", "error": str(exc)}), 503

@app.post("/api/upload")
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "Choose a CSV file to upload."}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file was selected."}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 415
    raw = file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "File exceeds the 30 MB upload limit."}), 413
    try:
        df = pd.read_csv(io.BytesIO(raw))
        mapping = infer_columns(df)
        result = _load_dataframe(df, secure_filename(file.filename), False)
        result.update({
            "column_mapping": mapping,
            "columns": list(df.columns),
            "preview": df.head(8).fillna("").astype(str).to_dict("records"),
        })
        return jsonify(json_safe(result))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.get("/api/summary")
def summary():
    return jsonify(_summary())

@app.get("/api/graph")
def graph():
    error = _require_graph()
    if error:
        return error
    try:
        limit = max(1, min(int(request.args.get("limit", 500)), 1000))
    except ValueError:
        return jsonify({"error": "limit must be an integer."}), 400
    G, T, C = STATE["G"], STATE["table"], STATE["communities"]
    ordered = sorted(G.nodes(), key=lambda n: T.loc[n, "interaction_volume"], reverse=True)[:limit]
    nodes = []
    for node in ordered:
        row = T.loc[node]
        nodes.append({
            "id": str(node), "label": str(node), "community": C.get(node, 0),
            "in_degree": int(row.in_degree), "out_degree": int(row.out_degree),
            "abuse_received_rate": float(row.abuse_received_rate),
            "abuse_sent_rate": float(row.abuse_sent_rate),
            "pagerank": float(row.pagerank), "volume": float(row.interaction_volume),
            "label_value": None if pd.isna(row.label) else int(row.label),
        })
    allowed = set(ordered)
    edges = [
        {"from": str(u), "to": str(v), "value": int(d["weight"]),
         "abusive": int(d["abusive_count"]), "label": f'{d["weight"]} interactions'}
        for u, v, d in G.edges(data=True) if u in allowed and v in allowed
    ]
    return jsonify({"nodes": nodes, "edges": edges, "truncated": G.number_of_nodes() > len(ordered)})

@app.get("/api/users")
def users():
    error = _require_graph()
    if error:
        return error
    q = request.args.get("q", "").strip().lower()
    try:
        limit = max(1, min(int(request.args.get("limit", 100)), 500))
    except ValueError:
        return jsonify({"error": "limit must be an integer."}), 400
    table = STATE["table"].copy()
    if q:
        table = table[table.index.to_series().str.lower().str.contains(q, regex=False)]
    table = table.sort_values("interaction_volume", ascending=False).head(limit)
    result = []
    for user_id, row in table.iterrows():
        item = {"user_id": str(user_id), "community": STATE["communities"].get(user_id)}
        item.update({k: (None if pd.isna(v) else float(v)) for k, v in row.items()})
        result.append(item)
    return jsonify(result)

@app.get("/api/communities")
def communities():
    error = _require_graph()
    if error:
        return error
    G, C, T = STATE["G"], STATE["communities"], STATE["table"]
    result = []
    for cid in sorted(set(C.values())):
        members = [node for node, community in C.items() if community == cid]
        subgraph = G.subgraph(members)
        result.append({
            "community": int(cid), "members": len(members),
            "internal_interactions": int(sum(d["weight"] for _, _, d in subgraph.edges(data=True))),
            "top_users": sorted(members, key=lambda n: T.loc[n, "interaction_volume"], reverse=True)[:6],
            "mean_abuse_received": float(T.loc[members, "abuse_received_rate"].mean()),
        })
    return jsonify(result)

@app.get("/api/targets")
def targets():
    error = _require_graph()
    if error:
        return error
    return jsonify({
        "repeated_targeting": json_safe(STATE["repeated"]),
        "temporal_bursts": json_safe(STATE["bursts"]),
        "interpretation": "Flags are evidence-based review candidates, not proof of coordinated abuse.",
    })

@app.post("/api/train")
def train():
    error = _require_graph()
    if error:
        return error
    try:
        epochs = int(request.args.get("epochs", 60))
        if epochs < 5 or epochs > 200:
            return jsonify({"error": "Choose between 5 and 200 epochs."}), 400
        results = train_models(STATE["G"], STATE["table"], STATE["X"], epochs)
        STATE["models"] = results
        save_model_run(STATE["dataset_id"], results)
        return jsonify(json_safe(results))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.get("/api/explain/<path:user_id>")
def explain(user_id):
    error = _require_graph()
    if error:
        return error
    return jsonify(json_safe(explain_user(
        user_id, STATE["G"], STATE["table"], STATE["communities"], STATE["repeated"]
    )))

@app.get("/api/report")
def report():
    error = _require_graph()
    if error:
        return error
    payload = {
        "project": "GNN-Based Harassment Network Analysis",
        "dataset": _summary(),
        "models": STATE["models"],
        "communities": communities().get_json(),
        "targets": targets().get_json(),
        "limitations": [
            "Dataset schema, labels and licence must be validated against source documentation.",
            "Community density and centrality do not establish abusive intent.",
            "Human review is required before moderation decisions.",
            "Random node split is a prototype protocol; graph-aware and temporal leakage checks are required.",
        ],
    }
    return Response(
        json.dumps(json_safe(payload), indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=gnn_analysis_report.json"},
    )

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Upload exceeds the configured 30 MB limit."}), 413

@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "An unexpected server error occurred. Check the terminal log."}), 500

def restore_latest_dataset():
    """Restore the latest persisted dataset after a server restart, if MySQL is configured."""
    try:
        record = load_latest_dataset()
        if record is not None:
            name, rows, dataset_id, is_demo = record
            df = pd.DataFrame(rows)
            _load_dataframe(df, name, is_demo, persist=False)
            STATE["dataset_id"] = dataset_id
    except Exception as exc:
        app.logger.warning("Could not restore previous dataset: %s", exc)

if __name__ == "__main__":
    init_database()
    restore_latest_dataset()
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "0") == "1")
