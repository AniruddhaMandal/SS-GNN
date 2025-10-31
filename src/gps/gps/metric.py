"""**Metric Builder**
Returns metric function `metric_fn`.
```python
def metric_fn(predict: Tensor(CPU), target: Tensor(CPU)) -> Dict
```
`metric_fn` returns a dict, e.g.`{'AP': 0.345}`
"""
import numpy as np
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

@register_metric('MRR')
def build_mrr():
    return _metric_decoretor(mean_reciprocal_rank, 'MRR')

def _metric_decoretor(func,name):
    def wrapper(*args, **kwargs):
        score = func(*args, **kwargs)
        return {name: score}
    return wrapper

def debug_predictions(edge_label, pred_logits, epoch):
    """Debug to see what your model is learning"""
    pred_probs = pred_logits
    labels = edge_label.cpu().numpy()
    
    print(f"\nEpoch {epoch} Debug:")
    print(f"Positive edges (label=1): mean_pred={pred_probs[labels==1].mean():.4f}")
    print(f"Negative edges (label=0): mean_pred={pred_probs[labels==0].mean():.4f}")
    
    # For MRR: Higher predictions should be positive edges
    if pred_probs[labels==1].mean() < pred_probs[labels==0].mean():
        print("⚠️ WARNING: Model predicting BACKWARDS!")

def mean_reciprocal_rank(target, predict):
    """
    Calculate Mean Reciprocal Rank (MRR) for binary relevance.
    
    Parameters:
    -----------
    target : numpy array of shape [num_edges_to_predict,]
        Binary array where 1 indicates relevant/positive edge, 0 otherwise
    predict : numpy array of shape [num_edges_to_predict,]
        Predicted scores for each edge (higher score = more likely)
    
    Returns:
    --------
    float : The Mean Reciprocal Rank score
    """
    # Sort indices by predicted scores in descending order
    #pred_probs = target 
    #labels = predict
    
    #print(f"\nEpoch {0} Debug:")
    #print(f"Positive edges (label=1): mean_pred={pred_probs[labels==1].mean():.4f}")
    #print(f"Negative edges (label=0): mean_pred={pred_probs[labels==0].mean():.4f}")
    
    ## For MRR: Higher predictions should be positive edges
    #if pred_probs[labels==1].mean() < pred_probs[labels==0].mean():
        #print("⚠️ WARNING: Model predicting BACKWARDS!")
        #exit()
    sorted_indices = np.argsort(-predict)
    
    # Get the sorted target values
    sorted_target = target[sorted_indices]
    
    # Find positions of all relevant items (1-indexed ranks)
    relevant_positions = np.where(sorted_target == 1)[0] + 1
    
    # MRR is the reciprocal of the first relevant item's rank
    if len(relevant_positions) > 0:
        mrr = 1.0 / relevant_positions[0]
    else:
        mrr = 0.0
    
    return mrr