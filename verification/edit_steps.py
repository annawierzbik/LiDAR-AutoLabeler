"""Step sizes of the box edit actions (web/service.py: forward, rotate_left, length_plus, ...),
in one place so they are defined once."""
import math

MOVE_STEP = 0.10              # m, forward/back/left/right
Z_STEP = 0.05                 # m, up/down
ROT_STEP = math.radians(2.0)  # rad, rotate left/right
SIZE_STEP = 0.10              # m, length/width/height -/+
