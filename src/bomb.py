
class Bomb:
    def __init__(self, data):
        self.row = data.get('row')
        self.col = data.get('col')
        self.player_id = data.get('playerId')
        self.remain_time = data.get('remainTime')

