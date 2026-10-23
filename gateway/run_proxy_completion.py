"""Track successful terminal completion separately from proxy stream termination."""

import json


class ProxyCompletion:
    """[DONE] closes a stream; only a successful terminal frame completes its turn."""

    def __init__(self):
        self.done = False
        self.stopped = False
        self.failed = False

    @property
    def completed(self) -> bool:
        return self.done and self.stopped and not self.failed

    def consume(self, line: str) -> tuple[bool, str]:
        line = line.strip()
        if not line.startswith("data: "):
            return False, ""
        data = line[6:]
        if data.strip() == "[DONE]":
            self.done = True
            return True, ""
        try:
            frame = json.loads(data)
            if frame.get("error"):
                self.failed = True
            flags = frame.get("hermes") or {}
            if (flags.get("failed") or flags.get("partial")
                    or flags.get("completed") is False or flags.get("error")):
                self.failed = True
            choices = frame.get("choices") or []
            choice = choices[0] if choices else {}
            content = choice.get("delta", {}).get("content", "")
            finish = choice.get("finish_reason")
            if finish is not None:
                self.stopped = finish == "stop"
                self.failed = self.failed or not self.stopped
            return False, content or ""
        except (json.JSONDecodeError, TypeError, AttributeError, IndexError):
            return False, ""
