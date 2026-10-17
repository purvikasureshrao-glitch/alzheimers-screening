"""Append-only run log for Colab / Kaggle sessions.

Colab loses its screen output when it disconnects. `RunLog` writes every message and every
`tee` block to a text file on the session disk (flushed immediately), so after a disconnect the log
still shows what finished, what its output was, and how long it took.
"""
import contextlib
import sys
import time
import traceback
from pathlib import Path


class RunLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log("---- session start ----")

    def _write(self, text):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text)
            f.flush()

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line)
        self._write(line + "\n")

    @contextlib.contextmanager
    def tee(self, title):
        """Run a block, copying everything it prints (and any error) into the log file."""
        log, orig, t0 = self, sys.stdout, time.time()

        class _Tee:
            def write(self, s):
                orig.write(s)
                log._write(s)
                return len(s)

            def flush(self):
                orig.flush()

        self._write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}  >>> {title}\n")
        sys.stdout = _Tee()
        try:
            yield
        except BaseException:
            self._write("!!! FAILED\n" + traceback.format_exc())
            raise
        finally:
            sys.stdout = orig
            self._write(f"<<< {title} done in {time.time() - t0:.1f}s\n")
