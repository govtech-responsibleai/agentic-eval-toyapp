"""In-memory model of the Redis memory bookkeeping used by the stores.

It records the sorted set and size hash that the stores write, and serves
`INFO memory`. It does not run the Lua eviction script: `eval` returns what a
test scripts, and the real script is covered by the opt-in live test.
"""

from collections.abc import Callable, Mapping


class FakeMemoryState:
    """Bookkeeping keys plus a scripted `INFO memory` and `EVAL`."""

    def __init__(self) -> None:
        self.lru: dict[str, float] = {}
        self.sizes: dict[str, str] = {}
        self.used_memory = 0
        self.maxmemory: int | None = 1 << 40
        self.eval_calls: list[tuple[str, ...]] = []
        self.evict: Callable[[tuple[str, ...]], object] = lambda _args: []

    async def info(self, section: str) -> Mapping[str, object]:
        assert section == "memory"
        info: dict[str, object] = {"used_memory": self.used_memory}
        if self.maxmemory is not None:
            info["maxmemory"] = self.maxmemory
        return info

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object:
        del script, numkeys
        self.eval_calls.append(keys_and_args)
        return self.evict(keys_and_args)

    def apply(self, command: str, arguments: tuple[object, ...]) -> object:
        """Apply one queued bookkeeping command; return its Redis reply."""
        if command == "zadd":
            mapping = arguments[1]
            assert isinstance(mapping, dict)
            for member, score in mapping.items():  # pyright: ignore[reportUnknownVariableType]
                self.lru[str(member)] = float(score)  # pyright: ignore[reportUnknownArgumentType]
            return 1
        if command == "hset":
            self.sizes[str(arguments[1])] = str(arguments[2])
            return 1
        if command == "zrem":
            return sum(self.lru.pop(str(m), None) is not None for m in arguments[1:])
        assert command == "hdel"
        return sum(self.sizes.pop(str(key), None) is not None for key in arguments[1:])
