#!/usr/bin/env python
# encoding: utf-8
"""
spoins

Copyright (c) 2019 __CGD Inc__. All rights reserved.
"""
from __future__ import absolute_import, unicode_literals


class SpoilType(object):
    REALITY_STONE = 5


class Spoil(object):
    def __init__(self, data):
        self.spoil_type = data.get('spoil_type')
        self.row = data.get('row')
        self.col = data.get('col')
