import os
import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path


class ResultsDB:
    """База данных результатов обучения моделей."""
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "..", "tuning", "results.db")
        self.db_path = str(Path(db_path).resolve())
        self._init_db()
    
    def _init_db(self):
        """Создаёт таблицы если нет."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                data_file TEXT NOT NULL,
                target TEXT NOT NULL,
                mode TEXT NOT NULL,
                score REAL,
                best_params TEXT,
                selected_features TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(run_name, model_name, data_file, target)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                trial_number INTEGER NOT NULL,
                value REAL,
                params TEXT,
                state TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_run_name(self, model_name, data_file, target):
        """Формирует имя прогона."""
        data_name = Path(data_file).stem if data_file else "unknown"
        return f"{model_name}_{data_name}_{target}"
    
    def start_run(self, model_name, data_file, target, mode="base"):
        """Начинает новый прогон или возвращает существующий."""
        run_name = self._get_run_name(model_name, data_file, target)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        try:
            cursor.execute("""
                INSERT INTO runs (run_name, model_name, data_file, target, mode, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (run_name, model_name, data_file, target, mode, now))
            run_id = cursor.lastrowid
            conn.commit()
            print(f"🆕 Created new run: {run_name} (id={run_id})")
        except sqlite3.IntegrityError:
            cursor.execute("""
                SELECT id FROM runs WHERE run_name = ? AND model_name = ? AND data_file = ? AND target = ?
            """, (run_name, model_name, data_file, target))
            run_id = cursor.fetchone()[0]
            print(f"📂 Using existing run: {run_name} (id={run_id})")
        
        conn.close()
        return run_id, run_name
    
    def save_score(self, run_id, score):
        """Сохраняет финальный score."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE runs SET score = ? WHERE id = ?
        """, (score, run_id))
        
        conn.commit()
        conn.close()
    
    def save_best_params(self, run_id, best_params):
        """Сохраняет лучшие параметры."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        params_json = json.dumps(best_params, default=str)
        
        cursor.execute("""
            UPDATE runs SET best_params = ? WHERE id = ?
        """, (params_json, run_id))
        
        conn.commit()
        conn.close()
    
    def save_selected_features(self, run_id, features):
        """Сохраняет выбранные признаки."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        features_json = json.dumps(list(features), default=str)
        
        cursor.execute("""
            UPDATE runs SET selected_features = ? WHERE id = ?
        """, (features_json, run_id))
        
        conn.commit()
        conn.close()
    
    def save_trial(self, run_id, trial_number, value, params, state="COMPLETE"):
        """Сохраняет информацию о trial."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        params_json = json.dumps(params, default=str)
        
        cursor.execute("""
            INSERT INTO trials (run_id, trial_number, value, params, state, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, trial_number, value, params_json, state, now))
        
        conn.commit()
        conn.close()
    
    def save_metric(self, run_id, metric_name, metric_value):
        """Сохраняет метрику."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO metrics (run_id, metric_name, metric_value, created_at)
            VALUES (?, ?, ?, ?)
        """, (run_id, metric_name, metric_value, now))
        
        conn.commit()
        conn.close()
    
    def get_run(self, model_name=None, data_file=None, target=None):
        """Получает список прогонов с фильтрацией."""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM runs WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if data_file:
            query += " AND data_file LIKE ?"
            params.append(f"%{Path(data_file).name}%")
        if target:
            query += " AND target = ?"
            params.append(target)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_all_runs(self):
        """Получает все прогоны."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM runs ORDER BY created_at DESC", conn)
        conn.close()
        return df
    
    def delete_run(self, run_id):
        """Удаляет прогон."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM trials WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        
        conn.commit()
        conn.close()


_results_db = None


def get_results_db():
    """Returns singleton ResultsDB."""
    global _results_db
    if _results_db is None:
        _results_db = ResultsDB()
    return _results_db