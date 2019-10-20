
class Position(object):
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __str__(self):
        return "Position(%s, %s)" % (self.row, self.col)

    def __repr__(self):
        return "Position(%s, %s)" % (self.row, self.col)


class Node:
    def __init__(self, row, col, dist=0, parent=None):
        self.row = row
        self.col = col
        self.parent = parent
        self.dist = dist

    def get_path(self):
        pth = [Position(self.row, self.col)]
        p = self.parent
        while p is not None:
            pth.append(Position(p.row, p.col))
            p = p.parent

        return list(reversed(pth))

