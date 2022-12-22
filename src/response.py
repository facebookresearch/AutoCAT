class Response:
    def __init__(self, hit_list, time, data=''):
        self.hit_list = hit_list
        self.time = time
        self.data = data

    def deepen(self, time, name):
        self.hit_list[name] = False
        self.time += time
