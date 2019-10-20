class Player:
    def __init__(self, data):
        self.id = data.get('id')
        current_position = data.get('currentPosition')

        if current_position:
            self.col = current_position.get('col')
            self.row = current_position.get('row')

        self.space_stone = data.get('spaceStone')
        self.mind_stone = data.get('mindStone')
        self.reality_stone = data.get('realityStone')
        self.power_stone = data.get('powerStone')
        self.time_stone = data.get('timeStone')
        self.soul_stone = data.get('soulStone')
