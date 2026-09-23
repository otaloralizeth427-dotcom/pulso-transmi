import os

PTM_API_BASE = os.environ.get("PTM_API_BASE", "https://pulso-transmi.72-60-245-2.sslip.io")
PTM_API_KEY = os.environ.get("PTM_API_KEY")

# Optional: only needed for traceability logging (pipeline_runs, predictions,
# model_state, validation_metrics). The pipeline still submits predictions to
# the competition without it -- it just skips local bookkeeping.
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL")

STATION_IDS = [
    "02300", "03000", "05000", "05100", "06000", "06111",
    "07105", "07107", "07111", "09000", "09122", "10009",
]
