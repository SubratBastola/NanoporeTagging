"""Job worker: `python -m nanotag.worker` (run by nanotag-worker.service).

Claims queued jobs from the database and runs up to NANOTAG_WORKERS of them in
parallel, each in its own process (so a crash or memory spike in one job cannot
take down the others or the web server).
"""
import logging
import multiprocessing as mp
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from .config import cfg
from .db import connect, migrate, now, one

log = logging.getLogger("nanotag.worker")
_stop = False


def _handle_term(signum, frame):
    global _stop
    _stop = True
    log.info("Stop requested; no new jobs will be started.")


def recover_interrupted():
    con = connect()
    try:
        con.execute("UPDATE jobs SET status='queued', started_at=NULL, "
                    "log=COALESCE(log,'') || '[requeued after worker restart]\n' "
                    "WHERE status='running' AND kind IN ('ingest','import_files','import_folder')")
        con.execute("UPDATE jobs SET status='error', finished_at=?, message='interrupted by worker restart' "
                    "WHERE status='running'", (now(),))
        con.execute("UPDATE cluster_runs SET status='error' WHERE status='running'")
        con.execute("UPDATE recordings SET status='pending' WHERE status='ingesting'")
    finally:
        con.close()


def claim_next():
    con = connect()
    try:
        con.execute("BEGIN IMMEDIATE")
        j = one(con, "SELECT id FROM jobs WHERE status='queued' ORDER BY id LIMIT 1")
        if j is None:
            con.execute("COMMIT")
            return None
        con.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (now(), j["id"]))
        con.execute("COMMIT")
        return j["id"]
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def _child(job_id):
    from .jobs import run_job
    run_job(job_id)


def main(poll_seconds=1.0, once=False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)
    c = cfg()
    c.ensure_dirs()
    migrate()
    recover_interrupted()
    n = max(1, c.WORKERS)
    log.info("nanotag worker %s starting with %d slot(s)", c.VERSION, n)
    ctx = mp.get_context("spawn")
    running = {}
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
        while not _stop:
            for jid, fut in list(running.items()):
                if fut.done():
                    exc = fut.exception()
                    if exc:
                        log.error("job %s crashed: %s", jid, exc)
                        con = connect()
                        con.execute("UPDATE jobs SET status='error', finished_at=?, message=? "
                                    "WHERE id=? AND status='running'", (now(), f"worker crash: {exc}"[:500], jid))
                        con.close()
                    running.pop(jid)
            while len(running) < n and not _stop:
                jid = claim_next()
                if jid is None:
                    break
                log.info("starting job %s", jid)
                running[jid] = pool.submit(_child, jid)
            if once and not running:
                break
            time.sleep(poll_seconds)
    log.info("worker stopped")


if __name__ == "__main__":
    once = "--once" in sys.argv
    main(once=once)
