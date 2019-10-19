
class Bomb:
    def __init__(self, data):
        self.row = data["row"]
        self.col = data["col"]
        self.player_id = data["playerId"]
        self.remain_time = data["remainTime"]

