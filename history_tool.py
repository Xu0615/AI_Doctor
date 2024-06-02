class Display_History(object):

    def __init__(self):
        super().__init__()
        self.history = []

    def add(self, first, second):
        self.history.append([first, second])

    def export(self):
        return self.history

    def clear(self):
        self.history = []