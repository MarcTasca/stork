from stork.nodes import ReadoutGroup


class NonLeakyReadoutGroup(ReadoutGroup):
    """Non-leaky accumulator readout used by periodic-reset networks.

    The constructor and state layout intentionally match ``ReadoutGroup`` so
    this group can replace the historical research-only non-leaky readout in
    existing Stork models.
    """

    def __init__(self, *args, gamma=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def forward(self):
        new_out = self.out + self.input * self.gamma
        self.out = self.states["out"] = new_out
