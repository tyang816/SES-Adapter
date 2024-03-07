from torchmetrics.classification import Accuracy, MulticlassMatthewsCorrCoef, F1Score, Recall


if __name__ == '__main__':
    
    
    
    metrics = {
        "acc": Accuracy(task="multiclass", num_classes=args.num_labels).to(device),
        "mcc": MulticlassMatthewsCorrCoef(num_classes=args.num_labels).to(device),
        "f1": F1Score(task="multiclass", num_classes=args.num_labels).to(device),
        "recall": Recall(task="multiclass", average='macro', num_classes=args.num_labels).to(device)
    }