from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import itertools

from .abc.base import LMScorer
from .gpt2 import GPT2LMScorer


class AutoLMScorer:
    MODEL_CLASSES = "pierreguillou/gpt2-small-portuguese"

    def __init__(self):
        raise EnvironmentError(
            "AutoLMscorer is designed to be instantiated "
            "using the `AutoLMscorer.from_pretrained(model_name)`"
            "method"
        )

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: Any) -> LMScorer:
        if 'gp2' not in model_name:
            raise ValueError(
                "Model name must be gpt2."
            )
        
        return GPT2LMScorer(model_name, **kwargs)

    @classmethod
    def supported_model_names(cls) -> Iterable[str]:
        classes = cls.MODEL_CLASSES
        models = map(lambda c: c.supported_model_names(), classes)
        return itertools.chain.from_iterable(models)
