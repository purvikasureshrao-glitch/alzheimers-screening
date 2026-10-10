"""The three Sarawgi et al. networks (PyTorch port of the official Keras models).

Layer sizes: official code (FINDINGS section 5). Porting Keras -> PyTorch is an
implementation choice [OURS] so every model exports cleanly to ONNX; the architecture
is unchanged. Block order Dense(ReLU) -> BN -> Dropout, regularizers and layer sizes
are copied from the official models.py (create_pause_model / create_compare_model /
create_intervention_model, read 2026-10-09).

Regularizers (Keras semantics), applied as loss penalties via `.penalty()`:
  kernel l2(0.01)    = 0.01 * sum(W^2)            (weights only, not bias)
  activity l1(0.01)  = 0.01 * mean_over_batch(sum |layer output|)
The activity l1 on a softmax output is constant (outputs sum to 1) so it has no gradient
and is omitted; it is kept only where it acts (the acoustic hidden ReLU layer).
"""
import torch
import torch.nn as nn

REG = 0.01


def _block(n_in, n_out, drop):
    return [nn.Linear(n_in, n_out), nn.ReLU(), nn.BatchNorm1d(n_out), nn.Dropout(drop)]


class DisfluencyNet(nn.Module):
    """11 features -> BN -> 16 -> 16 -> 24 -> 24 (dropout .2/.2/.1/.1) -> 2."""

    def __init__(self, n_in=11):
        super().__init__()
        layers = [nn.BatchNorm1d(n_in)]
        for a, b, d in [(n_in, 16, .2), (16, 16, .2), (16, 24, .1), (24, 24, .1)]:
            layers += _block(a, b, d)
        self.body, self.out = nn.Sequential(*layers), nn.Linear(24, 2)

    def forward(self, x):
        return self.out(self.body(x))  # logits; softmax applied outside

    def penalty(self):  # only the softmax layer is regularised in the official code
        return REG * (self.out.weight ** 2).sum()


class AcousticNet(nn.Module):
    """21 PCA components -> Dense 24 ReLU -> BN -> dropout .2 -> 2."""

    def __init__(self, n_in=21):
        super().__init__()
        self.fc, self.bn, self.drop = nn.Linear(n_in, 24), nn.BatchNorm1d(24), nn.Dropout(.2)
        self.out = nn.Linear(24, 2)
        self._act = None

    def forward(self, x):
        self._act = torch.relu(self.fc(x))  # kept for the activity l1 penalty
        return self.out(self.drop(self.bn(self._act)))

    def penalty(self):  # call after forward() on the same batch
        kernel = REG * ((self.fc.weight ** 2).sum() + (self.out.weight ** 2).sum())
        activity = REG * self._act.abs().sum(dim=1).mean()
        return kernel + activity


class InterventionsNet(nn.Module):
    """(32, 3) speaker sequence -> LSTM 16 -> BN -> dropout .2 -> Dense 16 ReLU -> BN -> 2."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 16, batch_first=True)
        self.bn1, self.drop = nn.BatchNorm1d(16), nn.Dropout(.2)
        self.fc, self.bn2, self.out = nn.Linear(16, 16), nn.BatchNorm1d(16), nn.Linear(16, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)  # last hidden state, as Keras LSTM(return_sequences=False)
        h = self.drop(self.bn1(h[-1]))
        return self.out(self.bn2(torch.relu(self.fc(h))))

    def penalty(self):  # no regularisers in the official intervention model
        return 0.


MODELS = {"disfluency": DisfluencyNet, "acoustic": AcousticNet, "interventions": InterventionsNet}
