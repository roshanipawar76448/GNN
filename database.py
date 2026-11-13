from __future__ import annotations

import json
import os
from datetime import datetime
import mysql.connector
from mysql.connector import Error
from pathlib import Path

                                                                              
                                                                       
def _load_env_file():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_env_file()

def _config(include_database=True):
    cfg = {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "autocommit": False,
    }
    if include_database:
        cfg["database"] = os.getenv("DB_NAME", "gnn_harassment")
    return cfg

def get_connection(include_database=True):
    return mysql.connector.connect(**_config(include_database))

def init_database():
    """Create the configured database/tables, or use a pre-created database."""
    db_name = os.getenv("DB_NAME", "gnn_harassment")
    if not db_name.replace("_", "").isalnum():
        raise ValueError("DB_NAME may contain only letters, numbers and underscores.")

                                                                             
                                                                                  
    server_connection = None
    try:
        server_connection = get_connection(False)
        server_cursor = server_connection.cursor()
        server_cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        server_connection.commit()
    except Error as create_error:
        try:
            probe = get_connection(True)
            probe.close()
        except Error as connect_error:
            raise RuntimeError(
                f"Cannot create or connect to MySQL database '{db_name}'. "
                "Create it using database/schema.sql or grant CREATE DATABASE permission. "
                f"MySQL details: {connect_error}"
            ) from create_error
    finally:
        if server_connection is not None:
            try:
                server_cursor.close()
            except Exception:
                pass
            server_connection.close()

    connection = get_connection(True)
    cursor = connection.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                dataset_name VARCHAR(255) NOT NULL,
                is_demo BOOLEAN NOT NULL DEFAULT FALSE,
                row_count INT NOT NULL,
                user_count INT NOT NULL,
                edge_count INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_datasets_created (created_at)
            ) ENGINE=InnoDB
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                dataset_id BIGINT NOT NULL,
                source_user VARCHAR(191) NOT NULL,
                target_user VARCHAR(191) NOT NULL,
                message_text TEXT,
                is_abusive TINYINT NULL,
                interaction_time DATETIME NULL,
                interaction_type VARCHAR(80) NULL,
                report_count DOUBLE NOT NULL DEFAULT 0,
                user_label TINYINT NULL,
                CONSTRAINT fk_interactions_dataset FOREIGN KEY (dataset_id)
                    REFERENCES datasets(id) ON DELETE CASCADE,
                INDEX idx_interactions_dataset (dataset_id),
                INDEX idx_interactions_source (dataset_id, source_user),
                INDEX idx_interactions_target (dataset_id, target_user),
                INDEX idx_interactions_time (dataset_id, interaction_time)
            ) ENGINE=InnoDB
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                dataset_id BIGINT NOT NULL,
                metrics_json LONGTEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT fk_model_dataset FOREIGN KEY (dataset_id)
                    REFERENCES datasets(id) ON DELETE CASCADE,
                INDEX idx_model_dataset (dataset_id)
            ) ENGINE=InnoDB
        """)
        connection.commit()
    finally:
        cursor.close()
        connection.close()

def save_dataset(name, interactions, is_demo):
    connection = get_connection()
    cursor = connection.cursor()
    try:
        nodes = set(interactions["source_user"].astype(str)) | set(interactions["target_user"].astype(str))
        edges = interactions[["source_user", "target_user"]].drop_duplicates().shape[0]
        cursor.execute(
            "INSERT INTO datasets(dataset_name,is_demo,row_count,user_count,edge_count) VALUES(%s,%s,%s,%s,%s)",
            (name[:255], bool(is_demo), int(len(interactions)), len(nodes), int(edges))
        )
        dataset_id = cursor.lastrowid
        rows = []
        for row in interactions.itertuples(index=False):
            ts = getattr(row, "timestamp", None)
            if ts is not None and str(ts) != "NaT":
                try:
                    ts = ts.to_pydatetime().replace(tzinfo=None) if hasattr(ts, "to_pydatetime") else ts
                except Exception:
                    ts = None
            else:
                ts = None
            abusive = getattr(row, "is_abusive", None)
            if abusive is not None:
                try:
                    if abusive != abusive:
                        abusive = None
                    else:
                        abusive = int(abusive)
                except Exception:
                    abusive = None
            label = getattr(row, "user_label", None)
            try:
                if label is not None and label == label:
                    label = int(label)
                else:
                    label = None
            except Exception:
                label = None
            rows.append((
                dataset_id, str(row.source_user)[:191], str(row.target_user)[:191],
                str(getattr(row, "text", ""))[:60000], abusive, ts,
                str(getattr(row, "interaction_type", "interaction"))[:80],
                float(getattr(row, "report_count", 0) or 0), label
            ))
        cursor.executemany("""
            INSERT INTO interactions
            (dataset_id,source_user,target_user,message_text,is_abusive,interaction_time,interaction_type,report_count,user_label)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, rows)
        connection.commit()
        return int(dataset_id)
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

def load_latest_dataset():
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id,dataset_name,is_demo FROM datasets WHERE is_demo=FALSE ORDER BY id DESC LIMIT 1")
        dataset = cursor.fetchone()
        if not dataset:
            return None
        cursor.execute("""
            SELECT source_user,target_user,message_text AS text,is_abusive,
                   interaction_time AS timestamp,interaction_type,report_count,user_label
            FROM interactions WHERE dataset_id=%s ORDER BY id
        """, (dataset["id"],))
        rows = cursor.fetchall()
        return dataset["dataset_name"], rows, int(dataset["id"]), bool(dataset["is_demo"])
    finally:
        cursor.close()
        connection.close()

def save_model_run(dataset_id, metrics):
    if dataset_id is None:
        return
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "INSERT INTO model_runs(dataset_id,metrics_json) VALUES(%s,%s)",
            (dataset_id, json.dumps(metrics, default=str))
        )
        connection.commit()
    finally:
        cursor.close()
        connection.close()

def get_db_status():
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT DATABASE()")
        name = cursor.fetchone()[0]
        return {"status": "connected", "database": name}
    finally:
        cursor.close()
        connection.close()
