"""Step sizes of the box edit keys, shared by the desktop window (verification/app.py) and the
browser (web/service.py) so the two UIs cannot drift apart."""
import math

MOVE_STEP = 0.10              # m, forward/back/left/right
Z_STEP = 0.05                 # m, up/down
ROT_STEP = math.radians(2.0)  # rad, rotate left/right
SIZE_STEP = 0.10              # m, length/width/height -/+
