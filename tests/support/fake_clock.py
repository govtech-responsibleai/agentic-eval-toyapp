"""Controllable clock for deterministic idle-order tests."""


class FakeClock:
    """Return a fixed time until a test advances it."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by `seconds`."""
        self.now += seconds
