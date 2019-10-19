class Node:
    def __init__(self, row, col, dist=0, parent=None):
        self.row = row
        self.col = col
        self.parent = parent
        self.dist = dist

    def get_path(self):
        pth = [(self.row, self.col)]
        p = self.parent
        while p is not None:
            pth.append((p.row, p.col))
            p = p.parent
        return pth

