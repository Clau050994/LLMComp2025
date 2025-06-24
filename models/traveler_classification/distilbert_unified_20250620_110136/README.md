# DistilBERT for Traveler Risk Assessment with Unified Dataset

This model is fine-tuned from `distilbert_robust_v2` with a comprehensive unified dataset to improve robustness across various challenging scenarios.

## Training Details
- Base model: models/traveler_classification/distilbert_robust_v2
- Fine-tuned at: 2025-06-20 11:07:57
- Training set size: 36528 examples from unified dataset
- Validation set: 5219 examples
- Learning rate: 5e-06
- Epochs: 8
- Custom loss: Medium-risk focus weighting

## Dataset Components

## Model Capabilities
This model has been specifically enhanced to handle:
1. **Confounding contexts** - Correctly identifying which country is relevant when multiple are mentioned
2. **Indirect descriptors** - Understanding risk levels from regional descriptions without explicit country names
3. **Mixed risk scenarios** - Processing complex cases with multiple countries or mixed signals
4. **Text variations** - Handling spelling variations, typos, and unusual formatting
5. **Medium risk identification** - Improved accuracy on medium-risk classification

## Performance Metrics

### Unified Test Set
{'eval_loss': 0.3517518639564514, 'eval_accuracy': 0.9074446680080482, 'eval_f1': 0.9074066659527401, 'eval_precision': 0.9075116618708761, 'eval_recall': 0.9074446680080482, 'eval_f1_low_risk': 0.9098320158102767, 'eval_f1_medium_risk': 0.9095956547978274, 'eval_f1_high_risk': 0.8991764967727576, 'eval_confusion_matrix': [[3683, 231, 109], [282, 3768, 80], [108, 156, 2020]], 'eval_runtime': 1.8272, 'eval_samples_per_second': 5712.125, 'eval_steps_per_second': 178.966, 'epoch': 7.996495838808585}

### Standard Clean Test Set
{'eval_loss': 0.040529411286115646, 'eval_accuracy': 0.9995, 'eval_f1': 0.9995000793479301, 'eval_precision': 0.999501014198783, 'eval_recall': 0.9995, 'eval_f1_low_risk': 0.9993021632937893, 'eval_f1_medium_risk': 1.0, 'eval_f1_high_risk': 0.9989847715736041, 'eval_confusion_matrix': [[716, 0, 1], [0, 791, 0], [0, 0, 492]], 'eval_runtime': 0.398, 'eval_samples_per_second': 5025.074, 'eval_steps_per_second': 158.29, 'epoch': 7.996495838808585}

### Noisy Test Set
{'eval_loss': 0.4224555492401123, 'eval_accuracy': 0.8835705045278137, 'eval_f1': 0.8833932998830666, 'eval_precision': 0.8835368472000504, 'eval_recall': 0.8835705045278137, 'eval_f1_low_risk': 0.8917819365337672, 'eval_f1_medium_risk': 0.889795918367347, 'eval_f1_high_risk': 0.8557993730407524, 'eval_confusion_matrix': [[1096, 80, 51], [96, 1090, 25], [39, 69, 546]], 'eval_runtime': 0.5857, 'eval_samples_per_second': 5278.964, 'eval_steps_per_second': 165.608, 'epoch': 7.996495838808585}

### Confounding Context Subset
{'eval_loss': 0.3516800105571747, 'eval_accuracy': 0.9054373522458629, 'eval_f1': 0.905402157763246, 'eval_precision': 0.9058863368542777, 'eval_recall': 0.9054373522458629, 'eval_f1_low_risk': 0.9090909090909091, 'eval_f1_medium_risk': 0.9037900874635568, 'eval_f1_high_risk': 0.9020689655172414, 'eval_confusion_matrix': [[585, 40, 11], [49, 620, 13], [17, 30, 327]], 'eval_runtime': 0.3457, 'eval_samples_per_second': 4894.299, 'eval_steps_per_second': 153.308, 'epoch': 7.996495838808585}

### Medium Risk Subset
{'eval_loss': 0.3851572573184967, 'eval_accuracy': 0.9123486682808717, 'eval_f1': 0.9541656115472271, 'eval_precision': 1.0, 'eval_recall': 0.9123486682808717, 'eval_f1_low_risk': 0.0, 'eval_f1_medium_risk': 0.9541656115472271, 'eval_f1_high_risk': 0.0, 'eval_confusion_matrix': [[0, 0, 0], [282, 3768, 80], [0, 0, 0]], 'eval_runtime': 0.7645, 'eval_samples_per_second': 5402.31, 'eval_steps_per_second': 170.049, 'epoch': 7.996495838808585}
