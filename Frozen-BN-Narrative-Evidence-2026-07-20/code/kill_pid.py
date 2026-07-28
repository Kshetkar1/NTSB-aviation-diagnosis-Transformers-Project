import os
import signal
import sys

pid = 36533
try:
    os.kill(pid, signal.SIGKILL)
    print(f"Successfully sent SIGKILL to {pid}")
except ProcessLookupError:
    print(f"Process {pid} does not exist")
except PermissionError:
    print(f"Permission denied to kill {pid}")
except Exception as e:
    print(f"Error: {e}")
