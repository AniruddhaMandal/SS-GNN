"""**Metric Builder**
Returns metric function `metric_fn`.
```python
def metric_fn(predict: Tensor(CPU), target: Tensor(CPU)) -> Dict
```
`metric_fn` returns a dict, e.g.`{'AP': 0.345}`
"""
from .registry import register_metric

@register_metric('AP')
def build_ap():
    from sklearn.metrics import average_precision_score
    metric_fn = average_precision_score
    return _metric_decoretor(metric_fn, 'AP')

@register_metric('ACC')
def build_acc():
    from sklearn.metrics import accuracy_score
    metric_fn = accuracy_score
    return _metric_decoretor(metric_fn, 'ACC')

@register_metric('MAE')
def build_mae():
    from sklearn.metrics import mean_absolute_error
    metric_fn = mean_absolute_error
    return _metric_decoretor(metric_fn, 'MAE')

def _metric_decoretor(func,name):
    def wrapper(*args, **kwargs):
        score = func(*args, **kwargs)
        return {name: score}
    return wrapper