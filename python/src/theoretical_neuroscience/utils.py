from typing import Dict

import numpy as np

D_TYPE = Dict[str, np.ndarray]


def postfix(x: Dict, txt):
    return {k + txt: v for k, v in x.items()}
