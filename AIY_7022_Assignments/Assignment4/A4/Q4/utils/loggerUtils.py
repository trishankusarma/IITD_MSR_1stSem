import sys
from datetime import datetime

class Logger:
    def __init__(self, filename):
        # If sys.stdout is already a Logger, unwrap to the real terminal
        if isinstance(sys.stdout, Logger):
            self.terminal = sys.stdout.terminal
        else:
            self.terminal = sys.stdout

        # Always start fresh
        self.log = open(filename, "w", buffering=1)

    def write(self, message):
        # Add timestamp prefix only for non-empty lines
        if message.strip():
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            message = ''.join(timestamp + line if line.strip() else line
                              for line in message.splitlines(True))

        # Write to real stdout and log file
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
