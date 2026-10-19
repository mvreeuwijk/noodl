"""Record the given golden cases with the code on PYTHONPATH; write JSON to argv[1]."""
import json
import sys

sys.path.insert(0, ".")
import noodl
from tests import test_golden_orchestration as G

out = {n: G.outcome(G.CASES[n]) for n in sys.argv[2:]}
json.dump({"noodl": noodl.__file__, "cases": json.loads(json.dumps(out))}, open(sys.argv[1], "w"))
print("recorded with", noodl.__file__)
