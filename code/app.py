import os
import time
from functools import wraps
from flask import Flask, render_template, request, jsonify

# Get the absolute path of the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__, 
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# --- SECURITY LAYER: IN-MEMORY RATE LIMITER ---
RATE_LIMIT = 5 # Maximum 5 requests
RATE_WINDOW = 60 # per 60 seconds
ip_records = {}

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        current_time = time.time()
        
        # Cleanup records older than the window
        if client_ip in ip_records:
            ip_records[client_ip] = [t for t in ip_records[client_ip] if current_time - t < RATE_WINDOW]
        else:
            ip_records[client_ip] = []
            
        # Check limit
        if len(ip_records[client_ip]) >= RATE_LIMIT:
            print(f"SECURITY ALERT: Rate limit exceeded by IP {client_ip}")
            return jsonify({"error": "Security Layer: Rate limit exceeded (Max 5 requests/min). Please slow down."}), 429
            
        ip_records[client_ip].append(current_time)
        return f(*args, **kwargs)
    return decorated_function

# --- SECURITY LAYER: HTTP HEADERS ---
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    # Restrict execution of malicious scripts while allowing necessary CDNs
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com;"
    return response

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Enterprise Sentiment Agent Web UI on port 5000...")
    app.run(debug=True, port=5000)
