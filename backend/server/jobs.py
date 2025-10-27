# backend/server/jobs.py
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional, List

_JOBS: Dict[str, dict] = {}

def _pump_stream(stream, log_file: Path):
    with log_file.open("a", encoding="utf-8", errors="ignore") as f:
        for line in iter(stream.readline, b""):
            try:
                f.write(line.decode("utf-8", errors="ignore"))
            except Exception:
                # fallback binaire
                f.write(line.decode("latin-1", errors="ignore"))
            f.flush()
    stream.close()

def start_job(cmd: list[str], logs_dir: Path, workdir: Path) -> str:
    logs_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    log_path = logs_dir / f"{job_id}.log"

    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        close_fds=False,
        shell=False,
    )

    t = threading.Thread(target=_pump_stream, args=(proc.stdout, log_path), daemon=True)
    t.start()

    _JOBS[job_id] = {
        "proc": proc,
        "log_path": log_path,
        "thread": t,
        "exit_code": None,
    }
    return job_id

def get_status(job_id: str) -> Optional[dict]:
    j = _JOBS.get(job_id)
    if not j:
        return None
    proc = j["proc"]
    if j["exit_code"] is None and proc.poll() is not None:
        j["exit_code"] = int(proc.returncode)
    running = proc.poll() is None
    return {"running": running, "exit_code": j["exit_code"]}

def read_logs(job_id: str, last_n: int = 200) -> Optional[List[str]]:
    j = _JOBS.get(job_id)
    if not j:
        # permettre lecture post-mortem
        # si le job a existé mais API relancée, tenter de lire sur disque
        # on accepte les lectures directes si fichier présent
        pass
    logs_dir = Path(__file__).resolve().parents[1] / "data" / "logs"
    log_path = logs_dir / f"{job_id}.log"
    if not log_path.exists():
        return None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if last_n > 0:
            lines = lines[-last_n:]
        return [l.rstrip("\n") for l in lines]
    except Exception:
        return []

