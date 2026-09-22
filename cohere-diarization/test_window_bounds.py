import numpy as np
from transcriber import _plan_window_bounds

mel = np.random.rand(1, 91765, 128).astype(np.float32)
b = _plan_window_bounds(mel, 12000)
assert b[0] == 0 and b[-1] == 91765, b
assert all(x < y for x, y in zip(b, b[1:])), "not monotonic"
gaps = [y - x for x, y in zip(b, b[1:])]
assert max(gaps) <= 12000, max(gaps)
assert min(gaps) >= 300, min(gaps)
assert _plan_window_bounds(np.zeros((1, 5000, 128), np.float32), 12000) == [0, 5000]

mel2 = np.ones((1, 24000, 128), np.float32)
mel2[0, 11950:12000, :] = 0.0
b2 = _plan_window_bounds(mel2, 12000)
assert 11950 <= b2[1] <= 12000, b2

print("ok", len(b), "bounds, max gap", max(gaps), "snap at", b2[1])
