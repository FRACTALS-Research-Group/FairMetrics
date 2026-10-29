"""
    Core metrics for the FairMetrics package.
"""
import torch


def combine_features(features):
    """
        Combine the separated features of each entry into a single custom feature.

        Parameters:
        - features: torch.Tensor
            The input feature matrix (num_of_features X num_of_entries).

        Returns:
        - torch.Tensor
            The combined feature matrix (num_of_entries X num_of_features).
    """
    if features.dim() == 1:
        features = torch.unsqueeze(features, 1)
    else:
        features = features.T
    return features


def demographic_parity(y_true, y_pred, sensitive_features):
    """
        Demographic parity metric.
        The same function implementation is used for both binary and multiclass classification.

        Parameters:
        - y_true: Any
            Unused, can be set to None;
            included to keep the function signature consistent with the others.
        - y_pred: torch.Tensor
            Predicted labels.
        - sensitive_features: torch.Tensor
            Sensitive feature matrix.
        
        Returns:
        - torch.Tensor
            Demographic parity metric.
    """
    unique_classes = torch.unique(y_pred)

    if len(unique_classes) == 1:
        return torch.tensor(0.0)
    elif len(unique_classes) == 2:
        unique_classes = torch.tensor([unique_classes[1]])

    sensitive_features = combine_features(sensitive_features)
    unique_groups = torch.unique(sensitive_features, dim=0)

    if len(unique_groups) < 2:
        return torch.tensor(0.0)

    max_diff = torch.tensor(0.0)
    for c in unique_classes:
        rates = []
        for group in unique_groups:
            class_mask = (y_pred == c)
            group_mask = (sensitive_features == group).sum(dim=1) == len(group)
            class_and_group_mask = ((class_mask * group_mask) == True)

            if (class_and_group_mask).sum() == 0:
                pos_pred_rate = torch.tensor(0.0)
            else:
                pos_pred_rate = (class_and_group_mask).sum().float() / (group_mask).sum().float()
            rates.append(pos_pred_rate)

        rates = torch.tensor(rates)
        diff = torch.max(rates) - torch.min(rates)
        max_diff = torch.max(max_diff, diff)

    return max_diff


def conditional_demographic_parity(y_true, y_pred, sensitive_features, conditional_features):
    """
        Conditional demographic parity metric.
        The same function implementation is used for both binary and multiclass classification.

        Parameters:
        - y_true: torch.Tensor
            True labels.
        - y_pred: torch.Tensor
            Predicted labels.
        - sensitive_features: torch.Tensor
            Sensitive feature matrix.
        - conditional_features: torch.Tensor
            Conditional feature matrix.
        
        Returns:
        - torch.Tensor
            Conditional demographic parity metric.
    """
    sensitive_features = combine_features(sensitive_features)
    conditional_features = combine_features(conditional_features)
    strata = torch.unique(conditional_features, dim=0)

    max_diff = torch.tensor(0.0)
    for s in strata:
        indices = (conditional_features == s).sum(dim=1) == len(s)
        y_pred_ = y_pred[indices]
        sens = sensitive_features[indices]
        strata_dp = demographic_parity(None, y_pred_, sens.T)
        max_diff = torch.max(max_diff, strata_dp)

    return max_diff


def diagonal_equality(y_true, y_pred, sensitive_features):
    """
        Diagonal equality metric.

        Parameters:
        - y_true: torch.Tensor
            True labels.
        - y_pred: torch.Tensor
            Predicted labels.
        - sensitive_features: torch.Tensor
            Sensitive feature matrix.

        Returns:
        - torch.Tensor
            Diagonal equality metric.
    """
    pred_unique_classes = torch.unique(y_pred)
    sensitive_features = combine_features(sensitive_features)
    unique_groups = torch.unique(sensitive_features, dim=0)

    if len(unique_groups) < 2:
        return torch.tensor(0.0)

    max_diff = torch.tensor(0.0)
    for c in pred_unique_classes:
        rates = []
        for group in unique_groups:
            group_mask = (sensitive_features == group).sum(dim=1) == len(group)
            y_true_and_group = ((y_true == c) & group_mask).float().sum()

            if y_true_and_group > torch.tensor(0.0):
                correct_predictions = (y_true == c) & (y_pred == c)
                correct_pred_and_group = (correct_predictions & group_mask).float().sum()
                rate = correct_pred_and_group / y_true_and_group
                rates.append(rate)

        if len(rates) >= 2:
            rates = torch.tensor(rates)
            diff = torch.max(rates) - torch.min(rates)
            max_diff = torch.max(max_diff, diff)

    return max_diff


def equal_opportunity(y_true, y_pred, sensitive_features):
    """
        Binary equal opportunity metric.

        Parameters:
        - y_true: torch.Tensor
            True labels.
        - y_pred: torch.Tensor
            Predicted labels.
        - sensitive_features: torch.Tensor
            Sensitive feature matrix.

        Returns:
        - torch.Tensor
            Equal opportunity metric.
    """
    sensitive_features = combine_features(sensitive_features)
    unique_groups = torch.unique(sensitive_features, dim=0)

    if len(unique_groups) < 2:
        return torch.tensor(0.0)

    true_positive = y_true == 1
    rates = []
    for group in unique_groups:
        group_mask = (sensitive_features == group).sum(dim=1) == len(group)
        true_pos_and_group_mask = (true_positive & group_mask)
        if true_pos_and_group_mask.sum() > torch.tensor(0):
            tpr = y_pred[true_pos_and_group_mask].float().mean()
            rates.append(tpr)

    if len(rates) == 0:
        return torch.tensor(0.0)

    rates = torch.tensor(rates)
    return torch.max(rates) - torch.min(rates)


def off_diagonal_equality(y_true, y_pred, sensitive_features):
    """
        Off-diagonal equality metric.

        Parameters:
        - y_true: torch.Tensor
            True labels.
        - y_pred: torch.Tensor
            Predicted labels.
        - sensitive_features: torch.Tensor
            Sensitive feature matrix.

        Returns:
        - torch.Tensor
            Off-diagonal equality metric.
    """
    pred_unique_classes = torch.unique(y_pred)
    sensitive_features = combine_features(sensitive_features)
    unique_groups = torch.unique(sensitive_features, dim=0)

    if len(unique_groups) < 2:
        return torch.tensor(0.0)

    max_diff = torch.tensor(0.0)
    for pred_c in pred_unique_classes:
        for true_c in pred_unique_classes:
            if pred_c == true_c:
                continue

            rates = []
            for group in unique_groups:
                group_mask = (sensitive_features == group).sum(dim=1) == len(group)
                y_true_and_group = ((y_true == true_c) & group_mask).float().sum()

                if y_true_and_group > torch.tensor(0.0):
                    wrong_labels_mask = ((y_true == true_c) & (y_pred == pred_c))
                    wrong_pred_and_group = (wrong_labels_mask & group_mask).float().sum()
                    rate = wrong_pred_and_group / y_true_and_group
                    rates.append(rate)

            if len(rates) >= 2:
                rates = torch.tensor(rates)
                diff = torch.max(rates) - torch.min(rates)
                max_diff = torch.max(max_diff, diff)

    return max_diff


def predictive_equality(y_true, y_pred, sensitive_features):
    """
        Predictive equality metric.

        Parameters:
        - y_true: torch.Tensor
            True labels.
        - y_pred: torch.Tensor
            Predicted labels.
        - sensitive_features: torch.Tensor
            Sensitive feature matrix.

        Returns:
        - torch.Tensor
            Predictive equality metric.
    """
    inverted_y_true = (y_true == 0).float()
    return equal_opportunity(inverted_y_true, y_pred, sensitive_features)


def equalized_odds(y_true, y_pred, sensitive_features):
    """
        Equalized odds metric.
        The same function implementation is used for both binary and multiclass classification.

        Parameters:
        - y_true: torch.Tensor
            True labels.
        - y_pred: torch.Tensor
            Predicted labels.
        - sensitive_features: torch.Tensor
            Sensitive feature matrix.

        Returns:
        - torch.Tensor
            Equalized odds metric.
    """
    return torch.max(
        diagonal_equality(y_true, y_pred, sensitive_features),
        off_diagonal_equality(y_true, y_pred, sensitive_features),
    )


def multiclass_predictive_parity(y_true, y_pred, sensitive_features):
    """
        Multiclass predictive parity metric.

        Parameters:
        - y_true: torch.Tensor
            True labels.
        - y_pred: torch.Tensor
            Predicted labels.
        - sensitive_features: torch.Tensor
            Sensitive feature matrix.

        Returns:
        - torch.Tensor
            Multiclass predictive parity metric.
    """
    true_unique_classes = torch.unique(y_true)
    sensitive_features = combine_features(sensitive_features)
    unique_groups = torch.unique(sensitive_features, dim=0)

    if len(unique_groups) < 2:
        return torch.tensor(0.0)

    max_diff = torch.tensor(0.0)
    for c in true_unique_classes:
        rates = []
        for group in unique_groups:
            group_mask = (sensitive_features == group).sum(dim=1) == len(group)
            y_pred_and_group = ((y_pred == c) & group_mask).float().sum()

            if y_pred_and_group > torch.tensor(0.0):
                correct_predictions = (y_true == c) & (y_pred == c)
                correct_pred_and_group = (correct_predictions & group_mask).float().sum()
                rate = correct_pred_and_group / y_pred_and_group
                rates.append(rate)

        if len(rates) >= 2:
            rates = torch.tensor(rates)
            diff = torch.max(rates) - torch.min(rates)
            max_diff = torch.max(max_diff, diff)

    return max_diff


def binary_predictive_parity(y_true, y_pred, sensitive_features):
    """
        Binary predictive parity metric.

        Parameters:
        - y_true: torch.Tensor
            True labels.
        - y_pred: torch.Tensor
            Predicted labels.
        - sensitive_features: torch.Tensor
            Sensitive feature matrix.

        Returns:
        - torch.Tensor
            Binary predictive parity metric.
    """
    sensitive_features = combine_features(sensitive_features)
    unique_groups = torch.unique(sensitive_features, dim=0)

    if len(unique_groups) < 2:
        return torch.tensor(0.0)

    pred_positive = y_pred == 1
    rates = []
    for group in unique_groups:
        group_mask = (sensitive_features == group).sum(dim=1) == len(group)
        pred_pos_and_group_mask = (pred_positive & group_mask)
        if pred_pos_and_group_mask.sum() > torch.tensor(0):
            rate = y_true[pred_pos_and_group_mask].float().mean()
            rates.append(rate)

    if len(rates) == 0:
        return torch.tensor(0.0)

    rates = torch.tensor(rates)
    return torch.max(rates) - torch.min(rates)


def equal_accuracy(y_true, y_pred, sensitive_features):
    """
        Equal accuracy metric.
        The same function implementation is used for both binary and multiclass classification.

        Parameters:
        - y_true: torch.Tensor
            True labels.
        - y_pred: torch.Tensor
            Predicted labels.
        - sensitive_features: torch.Tensor
            Sensitive feature matrix.

        Returns:
        - torch.Tensor
            Equal accuracy metric.
    """
    sensitive_features = combine_features(sensitive_features)
    unique_groups = torch.unique(sensitive_features, dim=0)

    if len(unique_groups) < 2:
        return torch.tensor(0.0)

    rates = []
    for group in unique_groups:
        group_mask = (sensitive_features == group).sum(dim=1) == len(group)
        group_accuracy = (y_pred[group_mask] == y_true[group_mask]).float().mean()
        rates.append(group_accuracy)

    rates = torch.tensor(rates)
    return torch.max(rates) - torch.min(rates)


__all__ = [
    "demographic_parity",
    "conditional_demographic_parity",
    "equal_opportunity",
    "diagonal_equality",
    "predictive_equality",
    "off_diagonal_equality",
    "equalized_odds",
    "binary_predictive_parity",
    "multiclass_predictive_parity",
    "equal_accuracy",
]
