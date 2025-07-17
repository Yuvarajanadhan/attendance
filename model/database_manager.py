import os
import sqlite3
from datetime import datetime, timedelta

class FaceDatabase:
    def __init__(self):
        """Initialize database with auto-created directory"""
        os.makedirs(os.path.join(os.path.dirname(__file__), "../recognized_logs"), exist_ok=True)
        db_path = os.path.join(os.path.dirname(__file__), "../recognized_logs/faces.db")
        
        try:
            self.conn = sqlite3.connect(db_path)
            self._create_table()
        except Exception as e:
            print(f"Database initialization failed: {str(e)}")
            raise

    def _create_table(self):
        """Create the database table if it doesn't exist"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS recognized_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                name TEXT NOT NULL,
                confidence FLOAT
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON recognized_faces(name)")
        self.conn.commit()

    def should_save_face(self, name):
        """Check if person hasn't been logged in the last hour"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT timestamp FROM recognized_faces 
                WHERE name = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (name,))
            
            if (last_record := cursor.fetchone()):
                last_time = datetime.strptime(last_record[0], "%Y-%m-%d %H:%M:%S")
                return (datetime.now() - last_time) >= timedelta(hours=1)
            return True  # No previous records
        except Exception as e:
            print(f"Cooldown check error: {str(e)}")
            return True  # Allow saving if check fails

    def log_face(self, name, confidence):
        """Safely log a face recognition"""
        try:
            if self.should_save_face(name):
                self.conn.execute(
                    "INSERT INTO recognized_faces (name, confidence) VALUES (?, ?)",
                    (name, confidence)
                )
                self.conn.commit()
                return True
            return False
        except Exception as e:
            print(f"Failed to log face: {str(e)}")
            return False

    def __del__(self):
        """Safe database connection cleanup"""
        if hasattr(self, 'conn'):
            self.conn.close()