class BaseLogger(object):
    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def write(self, s):
        raise NotImplementedError

class SimpleLogger(BaseLogger):
    def __init__(self, file_name):
        self.file_name = file_name

    def open(self):
        self.f = open(self.file_name, "w+")

    def close(self):
        self.f.close()

    def write(self, s):
        self.f.write(s)

