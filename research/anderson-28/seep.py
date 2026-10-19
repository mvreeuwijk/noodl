import pathlib
import sys
import tempfile

sys.path.insert(0, "C:/projects/noodl-ci3")
import torch

from tests.verification import test_sewer_losses_parity as T

tmp = pathlib.Path(tempfile.mkdtemp())
body = T.CHANNELS.replace("RECT_OPEN    0.8          0.9        0          0",
                          "RECT_CLOSED  0.8          0.9        0          0")
losses = ("\n[LOSSES]\nC1 0 0 0 NO 40\nC2 0 0 0 NO 25\nC3 0 0 0 NO 60\n"
          "\n[ADJUSTMENTS]\nCONDUCTIVITY    " + " ".join(["0.6"] * 12) + "\n")
res = T.compare(tmp, T.fixture(body=body, extra=losses), "seepage")
print("worst depth, flow margin, vol:", res[:3])
torch.save(res[4], sys.argv[1])
