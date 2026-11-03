"""CityNav bearings: north=0, east=90, clockwise, degrees."""
import math


def bearing(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    if math.hypot(dx, dy) < 1e-8:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360


def angular_error(a, b):
    return abs((a - b + 180) % 360 - 180)


def city_heading(pose):
    if len(pose) == 6:
        if math.hypot(pose[3], pose[4]) < 1e-8:
            return None
        yaw = math.atan2(pose[4], pose[3])
    elif len(pose) == 5:
        yaw = pose[3]
    else:
        raise ValueError('Expected CityNav xyz+direction or xyz+yaw+pitch')
    return (90 - math.degrees(yaw)) % 360
