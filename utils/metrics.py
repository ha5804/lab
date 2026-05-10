from sklearn.metrics import roc_auc_score

def get_image_auc(test_loader, model):
    y_true = []
    y_score = []

    source_dataset = test_loader
    if hasattr(test_loader, "get_loader"):
        test_loader = test_loader.get_loader()

    dataset = test_loader.dataset

    for img, label in test_loader:
        score, _ = model.predict_batch(img)

        for item_label in label:
            if hasattr(source_dataset, "is_anomaly_label"):
                y_true.append(int(source_dataset.is_anomaly_label(item_label)))
            elif hasattr(source_dataset, "get_classes"):
                label_name = source_dataset.get_classes()[int(item_label)]
                y_true.append(int(label_name != "good"))
            elif hasattr(dataset, "is_anomaly_label"):
                y_true.append(int(dataset.is_anomaly_label(item_label)))
            elif hasattr(dataset, "get_classes"):
                label_name = dataset.get_classes()[int(item_label)]
                y_true.append(int(label_name != "good"))
            else:
                y_true.append(int(item_label != 0))

        y_score.extend(score.detach().cpu().tolist())

    return roc_auc_score(y_true, y_score)