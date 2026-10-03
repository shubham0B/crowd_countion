"""
src/reporter.py — Comprehensive Crowd Analytics and Report Generator
"""

import os
import json
import time
import pandas as pd
from datetime import datetime

REPORT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "session_reports.json")

def save_report_entry(media_type: str, filename: str, total_count: int, avg_count: float, peak_count: int, min_count: int, duration_sec: float, frame_data: list = None):
    """Saves a crowd analysis report entry persistently to disk and session."""
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "media_type": media_type,
        "filename": filename,
        "total_count": int(total_count),
        "avg_count": round(float(avg_count), 1),
        "peak_count": int(peak_count),
        "min_count": int(min_count),
        "duration_sec": round(float(duration_sec), 1),
        "frame_count": len(frame_data) if frame_data else 1
    }
    
    records = load_all_reports()
    records.insert(0, entry) # latest first
    
    try:
        with open(REPORT_FILE, "w") as f:
            json.dump(records[:100], f, indent=2)
    except Exception:
        pass
    
    return entry

def load_all_reports():
    """Loads all historical detection reports."""
    if os.path.exists(REPORT_FILE):
        try:
            with open(REPORT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def generate_html_report(latest_entry: dict, all_entries: list = None) -> str:
    """Generates a professional, printable HTML Executive Crowd Analytics Report."""
    now_str = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>VisionAI Crowd Analytics Executive Report</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f8fafc; color: #1e293b; }}
  .header {{ border-bottom: 2px solid #0284c7; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }}
  .title {{ font-size: 26px; font-weight: 700; color: #0f172a; margin: 0; }}
  .subtitle {{ color: #64748b; font-size: 14px; margin-top: 5px; }}
  .card-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
  .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }}
  .card-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: #64748b; margin-bottom: 6px; }}
  .card-value {{ font-size: 28px; font-weight: 800; color: #0284c7; }}
  .section {{ background: white; padding: 25px; border-radius: 12px; margin-bottom: 25px; border: 1px solid #e2e8f0; }}
  .section-title {{ font-size: 18px; font-weight: 700; margin-bottom: 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
  th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #475569; }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; }}
  .badge-success {{ background: #dcfce7; color: #15803d; }}
  .badge-warning {{ background: #fef3c7; color: #b45309; }}
  .badge-danger {{ background: #fee2e2; color: #b91c1c; }}
  .footer {{ text-align: center; font-size: 12px; color: #94a3b8; margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 15px; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1 class="title">👥 VisionAI — College Crowd Analytics Report</h1>
    <div class="subtitle">Automated Computer Vision Detection & Occupancy Summary</div>
  </div>
  <div style="text-align: right; font-size: 13px; color: #64748b;">
    <strong>Generated on:</strong><br>{now_str}
  </div>
</div>

<div class="card-grid">
  <div class="card">
    <div class="card-label">Peak Crowd Count</div>
    <div class="card-value">{latest_entry.get('peak_count', 0)}</div>
  </div>
  <div class="card">
    <div class="card-label">Average People / Frame</div>
    <div class="card-value">{latest_entry.get('avg_count', 0)}</div>
  </div>
  <div class="card">
    <div class="card-label">Minimum Crowd Count</div>
    <div class="card-value">{latest_entry.get('min_count', 0)}</div>
  </div>
  <div class="card">
    <div class="card-label">Analyzed Media</div>
    <div class="card-value" style="font-size: 18px; line-height: 28px; word-break: break-all;">{latest_entry.get('filename', 'Session')}</div>
  </div>
</div>

<div class="section">
  <div class="section-title">📊 Executive Summary & Key Insights</div>
  <p><strong>Primary Observation:</strong> During the analysis of <code>{latest_entry.get('filename', 'video')}</code> ({latest_entry.get('duration_sec', 0)}s duration), the system recorded an average occupancy of <strong>{latest_entry.get('avg_count', 0)} people</strong> with a maximum density surge reaching <strong>{latest_entry.get('peak_count', 0)} individuals</strong> simultaneously.</p>
  <p><strong>Density Classification:</strong> {'<span class="badge badge-danger">High Density Congestion</span>' if latest_entry.get('peak_count', 0) > 40 else ('<span class="badge badge-warning">Moderate Density</span>' if latest_entry.get('peak_count', 0) > 15 else '<span class="badge badge-success">Low / Normal Flow</span>')}</p>
</div>

<div class="section">
  <div class="section-title">🕒 Recent Session Log History</div>
  <table>
    <thead>
      <tr>
        <th>Timestamp</th>
        <th>Source</th>
        <th>File Name</th>
        <th>Peak Count</th>
        <th>Average Count</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
"""
    history = all_entries or [latest_entry]
    for r in history[:10]:
        badge = "badge-danger" if r.get('peak_count', 0) > 40 else ("badge-warning" if r.get('peak_count', 0) > 15 else "badge-success")
        status_text = "High Crowd" if r.get('peak_count', 0) > 40 else ("Moderate" if r.get('peak_count', 0) > 15 else "Normal")
        html += f"""
      <tr>
        <td>{r.get('timestamp', '')}</td>
        <td>{r.get('media_type', 'Video')}</td>
        <td><strong>{r.get('filename', '')}</strong></td>
        <td>{r.get('peak_count', 0)}</td>
        <td>{r.get('avg_count', 0)}</td>
        <td><span class="badge {badge}">{status_text}</span></td>
      </tr>
"""
    html += """
    </tbody>
  </table>
</div>

<div class="footer">
  Report generated by VisionAI Crowd Counting & Detection System • Autonomous Computer Vision Pipeline
</div>

</body>
</html>
"""
    return html
