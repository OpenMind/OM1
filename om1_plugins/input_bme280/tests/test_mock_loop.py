# Author: Wanbogang (@Wanbogang)
import os, threading, io, sys, contextlib
os.environ["MOCK"]="1"
os.environ["POLL_MS"]="200"

from om1_plugins.input_bme280.plugin import main

def test_loop_runs_briefly():
    # Hentikan cepat supaya tidak loop terus
    t = threading.Timer(0.7, lambda: sys.exit(0))
    t.daemon = True; t.start()
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            main()
        except SystemExit:
            pass
