"""Gunicorn configuration for production deployment."""
import multiprocessing
import os

# Server socket
# Render sets PORT automatically, use it if available
port = int(os.environ.get('PORT', '10000'))
bind = f"0.0.0.0:{port}"
backlog = 2048

# Worker processes
workers = 1  # Use only 1 worker to reduce memory usage
worker_class = "sync"
worker_connections = 1000
timeout = 300  # Increase timeout to 5 minutes for model compilation
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "pneumonia_detector"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Memory optimization
max_requests = 1000  # Restart workers after this many requests to prevent memory leaks
max_requests_jitter = 50

