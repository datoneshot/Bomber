class Player:
    def __init__(self, data):
        self.id = data["id"]
        self.col = data["currentPosition"]["col"]
        self.row = data["currentPosition"]["row"]
        self.space_stone = data["spaceStone"]
        self.mind_stone = data["mindStone"]
        self.reality_stone = data["realityStone"]
        self.power_stone = data["powerStone"]
        self.time_stone = data["timeStone"]
        self.soul_stone = data["soulStone"]
