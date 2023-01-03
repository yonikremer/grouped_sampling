from typing import Union, Tuple, List, Dict

from torch import LongTensor

TokenIDS = Union[List[int], Tuple[int]]
CompletionDict = Dict[str, Union[str, LongTensor]]