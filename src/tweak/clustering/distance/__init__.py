from enum import Enum


class DistanceCalcStatus(str, Enum):
    OK = 'OK'
    EMPTY = 'EMPTY'
    ONLY = 'ONLY'


class NotSupportedDistanceCalcStatus(Exception):
    pass
