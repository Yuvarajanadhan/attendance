import sqlite3
from datetime import datetime
import os

def display_records(limit=10, min_confidence=0.0):
    """Display recognition records with filters"""
    db_path = os.path.join(os.path.dirname(__file__), "../recognized_logs/faces.db")
    
    if not os.path.exists(db_path):
        print("No database found! Have you run any recognitions yet?")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get filtered data
    cursor.execute("""
        SELECT id, timestamp, name, confidence 
        FROM recognized_faces 
        WHERE confidence >= ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (min_confidence, limit))

    print("\nRECOGNITION HISTORY")
    print("=" * 50)
    print(f"{'ID':<5}{'Time':<20}{'Name':<20}{'Confidence':>10}")
    print("-" * 50)
    
    for row in cursor.fetchall():
        # Format timestamp for better readability
        dt = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
        print(f"{row[0]:<5}{dt.strftime('%Y-%m-%d %H:%M'):<20}{row[2]:<20}{row[3]:>10.2f}")

    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Number of records to show")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence score")
    args = parser.parse_args()

    display_records(limit=args.limit, min_confidence=args.min_confidence)