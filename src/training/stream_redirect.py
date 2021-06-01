import sys
import os


class RedirectAllOutput(object):
    def __init__(self, stream=sys.stdout, file=os.devnull):
        self.stream = stream
        self.file = file

    def __enter__(self):
        self.stream.flush()
        self.fd = open(self.file, "a")
        self.dup_stream = os.dup(self.stream.fileno())
        os.dup2(self.fd.fileno(), self.stream.fileno())  # replaces stream

    def __exit__(self, type, value, traceback):
        os.dup2(self.dup_stream, self.stream.fileno())  # restores stream
        os.close(self.dup_stream)
        self.fd.close()
