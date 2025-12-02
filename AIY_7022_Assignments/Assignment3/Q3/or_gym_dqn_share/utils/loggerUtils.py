import sys
from datetime import datetime

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)  # line-buffered

    def write(self, message):
        # Add timestamp prefix only for non-empty lines
        if message.strip():
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            message = ''.join(timestamp + line if line.strip() else line
                              for line in message.splitlines(True))
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
