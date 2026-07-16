import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orbitzoo.dynamics.classes import Body
from orbitzoo.dynamics.constants import EARTH_RADIUS
from orbitzoo.interface.main import Interface


def test_should_draw_connection_respects_distance_and_visibility():
    body1 = Body({"name": "body1", "initial_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
    body2 = Body({"name": "body2", "initial_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})

    body1.position = np.array([EARTH_RADIUS + 1000.0, 0.0, 0.0])
    body2.position = np.array([EARTH_RADIUS + 1500.0, 0.0, 0.0])

    params = {"show": True, "distance": 2000.0}

    assert Interface.should_draw_connection(body1, body2, params) is True
    assert Interface.should_draw_connection(body1, body2, {"show": True, "distance": 1000.0}) is False
    assert Interface.should_draw_connection(body1, body2, {"show": False, "distance": 2000.0}) is False
