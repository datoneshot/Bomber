from src.player import Player
from src.bomb import Bomb
from src.spoils import Spoil

import numpy as np


class Board:
    def __init__(self, data):
        map_info = data.get('map_info') or {}
        size = map_info.get('size')
        if size:
            self.cols = size['cols']
            self.rows = size['rows']

        players = map_info.get('players') or []
        self.players = self.transform_class(players, Player)

        data_map = map_info.get('map') or []
        self.map = np.array(data_map)

        bombs = map_info.get('bombs') or []
        self.bombs = self.transform_class(bombs, Bomb)

        self.tag_name = data.get('tag')
        self.spoils = self.transform_class(map_info.get('spoils') or [], Spoil)

    @staticmethod
    def transform_class(data, cls):
        results = [cls(item) for item in data]
        return results

    def get_player(self, player_id):
        player = [p for p in self.players if p.id == player_id]
        return player[0] if player else None
