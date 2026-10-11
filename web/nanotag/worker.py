"""Job worker: `python -m nanotag.worker` (run by nanotag-worker.service).

Claims queued jobs from the database and runs up to NANOTAG_WORKERS of them in
parallel, each in its own process (so a crash or memory spike in one job cannot
take down the others or the web server).

Each job gets a fresh process that leads its own process group, so a job can be
stopped on its own: the web page marks it 'cancelling', the worker sees that on its
next poll, kills the job's process group (the job and anything it started), lets
jobs.finish_cancel() clean up partial results and marks the job 'cancelled'.
"""
import logging
import multiprocessing as mp
import os
import signal
import sys
import time

from .config import cfg
from .db import connect, migrate, now, one, rows

log = logging.getLogger("nanotag.worker")
_stop = False
STOP_GRACE = 25.0          # seconds the worker waits for running jobs when the service stops


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
        stopping = rows(con, "SELECT id FROM jobs WHERE status='cancelling'")
    finally:
        con.close()
    # stop requests that were pending when the worker went down: the process is gone, finish the cancel
    if stopping:
        from .jobs import finish_cancel
        for j in stopping:
            finish_cancel(j["id"], "stopped (worker restarted)")


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
    try:
        os.setpgrp()          # own process group: stopping the job also stops what it started
    except OSError:
        pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl+C on a foreground worker stops the worker, not jobs
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    from .jobs import run_job
    run_job(job_id)


def _kill(proc, grace=5.0):
    """Stop a job process and its process group: SIGTERM, then SIGKILL after `grace` seconds."""
    if proc.pid is None:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)       # the job leads its own group (see _child)
        except ProcessLookupError:
            if proc.is_alive():            # killed before it could create its group
                (proc.terminate if sig == signal.SIGTERM else proc.kill)()
        except OSError:
            (proc.terminate if sig == signal.SIGTERM else proc.kill)()
        proc.join(wait)
        if not proc.is_alive():
            # the group may still hold grandchildren (e.g. a pool the job started)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            return


def _mark_crash(jid, exitcode):
    con = connect()
    try:
        con.execute("UPDATE jobs SET status='error', finished_at=?, message=? WHERE id=? AND status='running'",
                    (now(), f"worker crash: job process exited with code {exitcode}"[:500], jid))
        job = one(con, "SELECT * FROM jobs WHERE id=?", (jid,))
    finally:
        con.close()
    if job and job["status"] == "error":
        try:
            from .db import jload
            from .jobs import _on_job_failed
            _on_job_failed(job, jload(job["params_json"], {}), RuntimeError(f"exit code {exitcode}"))
        except Exception:  # noqa: BLE001
            pass


def _stop_requests(running):
    """Ids of running jobs whose stop was requested from the web page."""
    if not running:
        return []
    con = connect()
    try:
        ids = list(running)
        q = ",".join("?" * len(ids))
        return [r["id"] for r in rows(con, f"SELECT id FROM jobs WHERE status='cancelling' AND id IN ({q})", ids)]
    finally:
        con.close()


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
    running = {}        # job id -> Process
    from .jobs import finish_cancel
    while not _stop:
        # 1) stop requests
        for jid in _stop_requests(running):
            proc = running.pop(jid)
            log.info("stopping job %s (pid %s)", jid, proc.pid)
            _kill(proc)
            finish_cancel(jid, "stopped by user")
        # 2) finished jobs
        for jid, proc in list(running.items()):
            if not proc.is_alive():
                proc.join()
                if proc.exitcode not in (0, None):
                    log.error("job %s crashed: exit code %s", jid, proc.exitcode)
                    _mark_crash(jid, proc.exitcode)
                running.pop(jid)
        # 3) new jobs
        while len(running) < n and not _stop:
            jid = claim_next()
            if jid is None:
                break
            log.info("starting job %s", jid)
            p = ctx.Process(target=_child, args=(jid,), name=f"nanotag-job-{jid}")
            p.start()
            running[jid] = p
        if once and not running:
            break
        time.sleep(poll_seconds)
    # service stopping: give running jobs a little time, then stop them (ingest/import jobs are re-queued
    # on the next start by recover_interrupted; others are marked interrupted)
    t_end = time.time() + STOP_GRACE
    for jid, proc in running.items():
        proc.join(max(0.0, t_end - time.time()))
    for jid, proc in running.items():
        if proc.is_alive():
            log.info("job %s still running at shutdown; stopping it", jid)
            _kill(proc, grace=2.0)
    log.info("worker stopped")


if __name__ == "__main__":
    once = "--once" in sys.argv
    main(once=once)
