"""AVDN bearings: north=0, east=90, clockwise, degrees."""
import math


def bearing(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    if math.hypot(dx, dy) < 1e-8:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360


def angular_error(a, b):
    return abs((a - b + 180) % 360 - 180)


def latlon_to_xy(point, origin):
    r = 6371000.0
    return [r * math.radians(point[1] - origin[1]) * math.cos(math.radians(origin[0])),
            r * math.radians(point[0] - origin[0])]
