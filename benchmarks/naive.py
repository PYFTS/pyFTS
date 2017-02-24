#!/usr/bin/python
# -*- coding: utf8 -*-

from pyFTS import fts


class Naive(fts.FTS):
    def __init__(self, name):
        super(Naive, self).__init__(1, "Naïve " + name)
        self.name = "Naïve Model"
        self.detail = "Naïve Model"

    def forecast(self, data):
        return [k for k in data]

