"""Bounded native KV telemetry, not conversation storage or persistence."""
from collections import deque
import threading


class KvStatus:
    PREFIXES = {"KVPOOL", "KVSESSION", "KVEVICT", "KVRESET", "KVINVALIDATE", "KVCACHE", "KVMOVE", "KVREPLAY", "KVEND"}

    def __init__(self):
        self.lock = threading.Lock()
        self.pool = {}
        self.sessions = {}
        self.events = deque(maxlen=256)
        self.sequence = 0
        self.missed_events = 0

    def consume(self, line):
        """Return (recognized, invalidated session id or None)."""
        fields = line.split()
        if not fields or fields[0] not in self.PREFIXES:
            return False, None
        kind = fields[0]
        data = {}
        for field in fields[1:]:
            key, sep, value = field.partition("=")
            if sep:
                data[key] = int(value) if value.isascii() and value.isdigit() else value
        with self.lock:
            if kind == "KVPOOL":
                self.pool = data
                return True, None
            session = data.get("session")
            if not isinstance(session, int) or session < 0:
                return True, None
            old = self.sessions.get(session, {})
            generation, epoch = data.get("generation", 0), data.get("epoch", 0)
            if not isinstance(generation, int) or not isinstance(epoch, int):
                return True, None
            if (generation, epoch) < (old.get("generation", 0), old.get("epoch", 0)):
                return True, None
            state = dict(old, **data)
            invalidated = None
            if kind in ("KVEVICT", "KVRESET", "KVINVALIDATE"):
                state.update(state="evicted" if kind == "KVEVICT" else "empty", tokens=0, bytes=0, reserved_tokens=0)
                invalidated = session
            if kind == "KVEVICT":
                sequence = data.get("sequence", 0)
                if isinstance(sequence, int) and sequence > self.sequence:
                    self.missed_events += max(0, sequence - self.sequence - 1)
                    self.sequence = sequence
                    self.events.append(dict(data))
            if kind in ("KVCACHE", "KVMOVE", "KVREPLAY"):
                state["state"] = "active"
            if kind == "KVMOVE" and isinstance(data.get("from"), int):
                source = data["from"]
                self.sessions[source] = dict(self.sessions.get(source, {}), state="empty", tokens=0, bytes=0, reserved_tokens=0)
            self.sessions[session] = state
            return True, invalidated

    def snapshot(self):
        with self.lock:
            return {"pool": dict(self.pool), "sessions": {str(s): dict(i) for s, i in self.sessions.items()},
                    "evictions": [dict(e) for e in self.events], "last_sequence": self.sequence,
                    "missed_events": self.missed_events}
