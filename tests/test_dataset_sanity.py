import torch
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix

def _collect_targets(loader):
    ys = []
    for b in loader:
        ys.append(b.y.detach().cpu().view(-1).long())
    return torch.cat(ys) if ys else torch.tensor([], dtype=torch.long)

@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_dataset_label_sanity(split_name, data_loaders):
    loader = data_loaders[split_name]
    y = _collect_targets(loader)

    assert y.numel() > 0, f"{split_name} split is empty!"
    assert y.dtype == torch.long, f"{split_name} labels must be LongTensor, got {y.dtype}"
    assert y.min().item() >= 0, f"{split_name} has negative labels!"
    # ENZYMES -> 6 classes (0..5). If you want generic, infer from train later.
    assert y.max().item() <= 5, f"{split_name} has labels above 5!"

    if split_name == "train":
        uniq = torch.unique(y).sort().values
        assert torch.equal(uniq, torch.arange(6)), \
            f"Train labels must be contiguous 0..5, got {uniq.tolist()}"

    if split_name == "test":
        y_train = _collect_targets(data_loaders["train"])
        assert set(y.tolist()).issubset(set(y_train.tolist())), \
            "Test has unseen classes not present in train!"

def test_eval_path_matches_torch_accuracy(exp, data_loaders):
    """Sanity: torch accuracy == sklearn accuracy on TEST using experiment's model."""
    model = exp.model
    model.eval()

    logit_list, target_list = [], []
    import torch
    with torch.no_grad():
        for b in data_loaders["test"]:
            inputs, labels = exp._unpack_batch(b)
            inputs = exp._to_device(inputs)
            labels = labels.to(exp.device)

            logits = model(*inputs)                 # shape [B, C]
            logit_list.append(logits.detach().cpu())
            target_list.append(b.y.detach().cpu().view(-1).long())

    all_logits = torch.cat(logit_list, dim=0)
    all_targets = torch.cat(target_list, dim=0)
    all_preds = all_logits.argmax(dim=1)

    torch_acc = (all_preds == all_targets).float().mean().item()

    y_true = all_targets.numpy().astype(int)
    y_pred = all_preds.numpy().astype(int)
    sk_acc = accuracy_score(y_true, y_pred)

    assert abs(torch_acc - sk_acc) < 1e-8, \
        f"torch acc {torch_acc:.6f} != sklearn acc {sk_acc:.6f}"

    # Optional diagnostics if you want pytest to print them on failure:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(6))
    print("Confusion matrix:\n", cm)
