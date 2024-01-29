from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    patience: int = 4
    delta: float = 0.0
    cnt: int = field(init=False, default_factory=lambda: 0)
    min_val: float = field(init=False, default_factory=lambda: float("inf"))
    invert: bool = False

    def __call__(self, metric):
        metric = -metric if self.invert else metric
        if metric < self.min_val:
            self.min_val = metric
            self.cnt = 0
            print("Reset")
            return False

        if metric >= (self.min_val + self.delta):
            print("increase")
            self.cnt += 1

        return self.cnt >= self.patience
