"""GroqFlow — Multi-Tenant AI Knowledge Assistant & Autonomous Appointment Booking Engine."""

import os

# Limit OpenMP threads to avoid contention in single-container deployments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
