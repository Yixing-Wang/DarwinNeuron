"""
DB helpers aligned with the NEW schema:

Tables:
- algorithm(id, name, HyperparameterProblem, NonDefaultParameter)
- termination(id, xtol, cvtol, ftol, period, n_max_gen, n_max_evals, time)
- runs(id, problem_type, problem_id, algorithm_id, termination_id, batch_size, seeds_json, multi_run, save_history, verbose, result_filename)

Notes:
- SQLite does not have arrays/booleans -> store arrays as JSON, booleans as INTEGER (0/1).
- We add config_hash for algorithm & termination to dedupe rows.
- Polymorphic FK (problem_type -> nn_problems/bbob_problems) is validated with triggers.
"""

from __future__ import annotations
import os
import json
import sqlite3
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import random, time as _time

DEFAULT_DB = os.environ.get("LANDSCAPE_DB", "../../data/landscape-analysis.db")

# ---------------- utils ----------------

def _stable_json(d: Optional[dict]) -> str:
    return json.dumps(d or {}, sort_keys=True, separators=(",", ":"))

def _hash_cfg(d: dict) -> str:
    return hashlib.sha1(_stable_json(d).encode()).hexdigest()

def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    db_path = db_path or DEFAULT_DB
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=60000;")  # 60s
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def _table_cols(conn: sqlite3.Connection, table: str):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return {r[1]: (r[0], r[2]) for r in cur.fetchall()}

def _ensure_column(conn: sqlite3.Connection, table: str, col_def: str):
    name = col_def.split()[0]
    cols = _table_cols(conn, table)
    if name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")

def _exec_with_retry(cur: sqlite3.Cursor, sql: str, params=(),
                     tries: int = 8, base: float = 0.2, cap: float = 5.0):
    
    for i in range(tries):
        try:
            cur.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                delay = min(cap, base * (2 ** i)) + random.random() * 0.1
                _time.sleep(delay)
                continue
            raise


# ---------------- DDL ----------------

def ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # algorithm
    cur.execute("""
    CREATE TABLE IF NOT EXISTS algorithm (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      HyperparameterProblem INTEGER DEFAULT 0,
      NonDefaultParameter TEXT DEFAULT '{}',
      config_hash TEXT UNIQUE
    );
    """)

    # termination
    cur.execute("""
    CREATE TABLE IF NOT EXISTS termination (
      id INTEGER PRIMARY KEY,
      xtol REAL,
      cvtol REAL,
      ftol REAL,
      period INTEGER,
      n_max_gen INTEGER,
      n_max_evals INTEGER,
      time TEXT,
      config_hash TEXT UNIQUE
    );
    """)

    # runs (polymorphic FK via triggers)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS runs (
      id INTEGER PRIMARY KEY,
      problem_type TEXT NOT NULL,           -- 'nn' | 'bbob'
      problem_id INTEGER NOT NULL,          -- validated by triggers
      algorithm_id INTEGER NOT NULL REFERENCES algorithm(id) ON DELETE RESTRICT,
      termination_id INTEGER NOT NULL REFERENCES termination(id) ON DELETE RESTRICT,
      batch_size INTEGER NOT NULL,
      seeds_json TEXT NOT NULL,             -- JSON array of ints
      multi_run INTEGER DEFAULT 0,
      save_history INTEGER DEFAULT 1,
      verbose INTEGER DEFAULT 0,
      result_filename TEXT NOT NULL,
      created_at TEXT DEFAULT (datetime('now')),
      updated_at TEXT
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_problem ON runs(problem_type, problem_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_algo ON runs(algorithm_id);")

    # polymorphic validation triggers
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS trg_runs_check_problem_insert
    BEFORE INSERT ON runs
    FOR EACH ROW
    BEGIN
      SELECT
        CASE
          WHEN NEW.problem_type='nn' AND NOT EXISTS(SELECT 1 FROM nn_problems WHERE id=NEW.problem_id)
            THEN RAISE(ABORT, 'runs.problem_id not found in nn_problems')
        END;
      SELECT
        CASE
          WHEN NEW.problem_type='bbob' AND NOT EXISTS(SELECT 1 FROM bbob_problems WHERE id=NEW.problem_id)
            THEN RAISE(ABORT, 'runs.problem_id not found in bbob_problems')
        END;
    END;
    """)
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS trg_runs_check_problem_update
    BEFORE UPDATE OF problem_type, problem_id ON runs
    FOR EACH ROW
    BEGIN
      SELECT
        CASE
          WHEN NEW.problem_type='nn' AND NOT EXISTS(SELECT 1 FROM nn_problems WHERE id=NEW.problem_id)
            THEN RAISE(ABORT, 'runs.problem_id not found in nn_problems')
        END;
      SELECT
        CASE
          WHEN NEW.problem_type='bbob' AND NOT EXISTS(SELECT 1 FROM bbob_problems WHERE id=NEW.problem_id)
            THEN RAISE(ABORT, 'runs.problem_id not found in bbob_problems')
        END;
    END;
    """)

    conn.commit()


# -------------- Dataclass model--------------

@dataclass
class AlgorithmRow:
    name: str
    hyperparameter_problem: bool = False
    non_default_params: Optional[Dict[str, Any]] = None

    @classmethod
    def lookup_by_id(cls, id: int, db_path: str = DEFAULT_DB) -> "AlgorithmRow":
        con = connect(db_path); cur = con.cursor()
        cur.execute("SELECT name, HyperparameterProblem, NonDefaultParameter FROM algorithm WHERE id=?", (id,))
        row = cur.fetchone(); con.close()
        if not row:
            raise ValueError(f"algorithm id={id} not found.")
        return cls(name=row[0],
                   hyperparameter_problem=bool(row[1]),
                   non_default_params=json.loads(row[2] or "{}"))

    def _row_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "HyperparameterProblem": int(bool(self.hyperparameter_problem)),
            "NonDefaultParameter": _stable_json(self.non_default_params),
        }
        
    def write_to_db(self, db_path: str = DEFAULT_DB) -> int:
        con = connect(db_path); ensure_tables(con)
        row = self._row_dict()
        cfg_hash = _hash_cfg(row)
        cur = con.cursor()
        cur.execute("SELECT id FROM algorithm WHERE config_hash=?", (cfg_hash,))
        got = cur.fetchone()
        if got:
            con.close(); return got[0]
        _exec_with_retry(cur, """
            INSERT INTO algorithm (name, HyperparameterProblem, NonDefaultParameter, config_hash)
            VALUES (?,?,?,?)
        """, (self.name, row["HyperparameterProblem"], row["NonDefaultParameter"], cfg_hash))
        con.commit(); out = cur.lastrowid; con.close()
        return out


@dataclass
class TerminationRow:
    xtol: Optional[float] = None
    cvtol: Optional[float] = None
    ftol: Optional[float] = None
    period: Optional[int] = None
    n_max_gen: Optional[int] = None
    n_max_evals: Optional[int] = None
    time: Optional[str] = None  # ISO 或秒数字符串
    n_max_epochs: Optional[int] = None

    @classmethod
    def lookup_by_id(cls, id: int, db_path: str = DEFAULT_DB) -> "TerminationRow":
        con = connect(db_path)
        ensure_tables(con)
        cur = con.cursor()
        cur.execute("""
            SELECT xtol, cvtol, ftol, period, n_max_gen, n_max_evals, time, n_max_epochs
            FROM termination WHERE id = ?
        """, (id,))
        row = cur.fetchone()
        con.close()
        if not row:
            raise ValueError(f"termination id={id} not found.")
        return cls(*row)

    def _row_dict(self) -> Dict[str, Any]:
        return {
            "xtol": self.xtol,
            "cvtol": self.cvtol,
            "ftol": self.ftol,
            "period": self.period,
            "n_max_gen": self.n_max_gen,
            "n_max_evals": self.n_max_evals,
            "time": self.time,
            "n_max_epochs": self.n_max_epochs,
        }

    def to_pymoo_termination_tuple(self) -> Tuple[str, int]:
        if self.n_max_gen is not None:
            return ("n_gen", int(self.n_max_gen))
        if self.n_max_evals is not None:
            return ("n_eval", int(self.n_max_evals))
        if self.time is not None:
            try:
                secs = float(self.time)
                return ("time", secs)
            except Exception:
                pass
        # fallback
        return ("n_gen", 30)

    def to_pymoo_termination_for_problem(
        self, problem=None, batches_per_epoch: Optional[int] = None
    ) -> Tuple[str, int]:
        """
        prioritize epoch：if given n_max_epochs, batches_per_epoch set to ('n_gen', n_max_epochs * bpe)；
        else use gen
        """
        if self.n_max_epochs is not None:
            bpe = batches_per_epoch
            if bpe is None and problem is not None:
                bpe = getattr(problem, "batches_per_epoch", None)
            if bpe is None or int(bpe) <= 0:
                raise ValueError("batches_per_epoch is required to convert n_max_epochs -> n_gen")
            return ("n_gen", int(self.n_max_epochs) * int(bpe))

        return self.to_pymoo_termination_tuple()

    def write_to_db(self, db_path: str = DEFAULT_DB) -> int:
        con = connect(db_path); ensure_tables(con)
        row = self._row_dict()
        cfg_hash = _hash_cfg(row)
        cur = con.cursor()
        cur.execute("SELECT id FROM termination WHERE config_hash = ?", (cfg_hash,))
        got = cur.fetchone()
        if got:
            con.close(); return got[0]
        _exec_with_retry(cur, """
            INSERT INTO termination
                (xtol, cvtol, ftol, period, n_max_gen, n_max_evals, time, n_max_epochs, config_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (row["xtol"], row["cvtol"], row["ftol"], row["period"],
              row["n_max_gen"], row["n_max_evals"], row["time"], row["n_max_epochs"], cfg_hash))
        con.commit(); out = cur.lastrowid; con.close()
        return out



@dataclass
class RunRow:
    problem_type: str                 # 'nn' | 'bbob'
    problem_id: int
    algorithm_id: int
    termination_id: int
    batch_size: int
    seeds: list[int]
    multi_run: bool = False
    save_history: bool = True
    verbose: bool = False
    result_filename: str = "result.json"

    @classmethod
    def lookup_by_id(cls, id: int, db_path: str = DEFAULT_DB) -> "RunRow":
        con = connect(db_path); cur = con.cursor()
        cur.execute("""
            SELECT problem_type,problem_id,algorithm_id,termination_id,batch_size,
                   seeds_json,multi_run,save_history,verbose,result_filename
            FROM runs WHERE id=?""", (id,))
        row = cur.fetchone(); con.close()
        if not row:
            raise ValueError(f"runs id={id} not found.")
        return cls(
            problem_type=row[0], problem_id=row[1], algorithm_id=row[2], termination_id=row[3],
            batch_size=row[4], seeds=json.loads(row[5] or "[]"),
            multi_run=bool(row[6]), save_history=bool(row[7]), verbose=bool(row[8]),
            result_filename=row[9]
        )

    def write_to_db(self, db_path: str = DEFAULT_DB) -> int:
        con = connect(db_path); ensure_tables(con)
        cur = con.cursor()
        _exec_with_retry(cur, """
            INSERT INTO runs(problem_type,problem_id,algorithm_id,termination_id,batch_size,
                             seeds_json,multi_run,save_history,verbose,result_filename)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (self.problem_type, self.problem_id, self.algorithm_id, self.termination_id,
              int(self.batch_size), json.dumps(list(self.seeds or [])),
              int(bool(self.multi_run)), int(bool(self.save_history)),
              int(bool(self.verbose)), self.result_filename))
        con.commit(); out = cur.lastrowid; con.close()
        return out

    @staticmethod
    def touch_result(run_id: int, db_path: str = DEFAULT_DB):
        con = connect(db_path); cur = con.cursor()
        _exec_with_retry(cur, "UPDATE runs SET updated_at=datetime('now') WHERE id=?", (run_id,))
        con.commit(); con.close()