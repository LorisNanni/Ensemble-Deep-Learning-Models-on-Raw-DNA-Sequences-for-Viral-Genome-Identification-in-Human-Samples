from sklearn.metrics import roc_curve, auc

# =============== #
# SCORE FUNCTIONS #
# =============== #

def auroc(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auroc = auc(fpr, tpr)
    return auroc, fpr, tpr