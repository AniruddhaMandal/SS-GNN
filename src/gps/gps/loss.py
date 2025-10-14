"""**Loss/Criterion Builder**:
Rerturns a callable `loss_fn`. 
```python
def loss_fn(predict: Tesnsor, target: Tensor)-> Tensor
```
"""
from .registry import register_loss

@register_loss('BCEWithLogitsLoss')
def build_bcelogit():
    from torch.nn import BCEWithLogitsLoss
    return BCEWithLogitsLoss()

@register_loss("CrossEntropyLoss")
def build_crs_entpy():
    from torch.nn import CrossEntropyLoss
    return CrossEntropyLoss()
