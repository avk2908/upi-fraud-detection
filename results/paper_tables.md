## Table 1. Feature dimensions and configuration

| Configuration | Dimensions | Details |
| --- | --- | --- |
| Behavioral input | 18 | XGBoost; Isolation Forest |
| LSTM sequence | 5 × 6 | amount, type, balance errors, amount deviation, transaction gap |
| HGNN | account: 7; transaction: 18 | PaySim account and transaction entities |

## Table 2. Strong baselines

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced-Accuracy | TN | FP | FN | TP | Threshold | TrainSeconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.0 | 0.0 | 0.0 | 0.9274040606871932 | 0.0880298017526756 | 0.5 | 2988 | 0 | 12 | 0 | 0.9999999988190212 | 0.0764974000048823 |
| Random Forest | 1.0 | 0.8333333333333334 | 0.9090909090909092 | 1.0 | 1.0 | 0.9166666666666669 | 2988 | 0 | 2 | 10 | 0.5933333333333334 | 3.922910099994624 |
| XGBoost | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 2988 | 0 | 0 | 12 | 0.9312365055084229 | 0.2180845999973826 |
| Full fused model | 1.0 | 0.3333333333333333 | 0.5 | 0.9994701026327532 | 0.8847666222666222 | 0.6666666666666666 | 2988 | 0 | 8 | 4 | 0.7118334991725848 |  |
| Isolation Forest only | 0.0 | 0.0 | 0.0 | 0.5287539045069165 | 0.0042837845826247 | 0.4713855421686747 | 2817 | 171 | 12 | 0 | 0.6037956703037952 |  |
| LSTM only (FedAvg) | 0.0 | 0.0 | 0.0 | 0.598086791610888 | 0.0262084912900918 | 0.4994979919678715 | 2985 | 3 | 12 | 0 | 0.4774109721183777 |  |
| HGNN only | 0.0172413793103448 | 0.5 | 0.0333333333333333 | 0.7230031236055333 | 0.0119398586825567 | 0.6927710843373494 | 2646 | 342 | 6 | 6 | 1.0 |  |

## Table 3. Ablation study

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced-Accuracy | TN | FP | FN | TP | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full: XGB + IF + LSTM(FedAvg) + HGNN | 1.0 | 0.3333333333333333 | 0.5 | 0.9994701026327532 | 0.8847666222666222 | 0.6666666666666666 | 2988 | 0 | 8 | 4 | 0.7118334991725848 |
| Ablation: - Isolation Forest | 1.0 | 0.4166666666666667 | 0.5882352941176471 | 0.9994422132976348 | 0.8743499555999557 | 0.7083333333333334 | 2988 | 0 | 7 | 5 | 0.6877050079978839 |
| Ablation: - LSTM | 1.0 | 0.5 | 0.6666666666666666 | 0.9995816599732262 | 0.9026848151848154 | 0.75 | 2988 | 0 | 6 | 6 | 0.5966230928055754 |
| Ablation: - HGNN | 1.0 | 0.9166666666666666 | 0.9565217391304348 | 1.0 | 1.0 | 0.9583333333333331 | 2988 | 0 | 1 | 11 | 0.4963554647940558 |
| Ablation: - Federated Training | 1.0 | 0.5 | 0.6666666666666666 | 0.9995816599732262 | 0.9026848151848152 | 0.75 | 2988 | 0 | 6 | 6 | 0.7343924937412618 |

## Table 4. Centralized vs federated LSTM

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced-Accuracy | TN | FP | FN | TP | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Federated LSTM | 0.0 | 0.0 | 0.0 | 0.598086791610888 | 0.0262084912900918 | 0.4994979919678715 | 2985 | 3 | 12 | 0 | 0.4774109721183777 |
| Centralized LSTM | 0.0 | 0.0 | 0.0 | 0.7380075858991522 | 0.0464384639542849 | 0.4998326639892905 | 2987 | 1 | 12 | 0 | 0.8077132105827332 |

## Table 5. Temporal evaluation

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced-Accuracy | TN | FP | FN | TP | Threshold | train_rows | validation_rows | future_test_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chronological XGBoost | 0.0006673340006673 | 1.0 | 0.0013337779259753 | 1.0 | 1.0 | 0.5005003335557038 | 3 | 2995 | 0 | 2 | 0.0091082295402884 | 14000 | 3000 | 3000 |

## Table 6. Robustness stress tests

| subset_size | positive_count | precision | recall | f1 | pr_auc | Scenario |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 0 | 0.0 | 0.0 | 0.0 |  | high_value |
| 3000 | 12 | 1.0 | 0.3333333333333333 | 0.5 | 0.8847666222666222 | novel_beneficiary |
| 30 | 0 | 0.0 | 0.0 | 0.0 |  | extreme_amount_deviation |

## Table 7. Computational and communication overhead

| model | training_seconds | inference_seconds | inference_ms_per_transaction | model_bytes | feature_dimension | communication_upload_bytes | communication_download_bytes | communication_total_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.3411304000037489 | 0.0146257000014884 | 0.0024376166669147 | 100738 | 18 |  |  |  |
| Isolation Forest | 0.5762944000016432 | 0.09049309999682 | 0.0150821833328033 | 2029525 | 18 |  |  |  |
| Federated LSTM | 6.496698100003414 |  |  | 130466 | 6 | 1891260.0 | 1891260.0 | 3782520.0 |
| Centralized LSTM | 5.092989400000079 |  |  | 130498 | 6 |  |  |  |
| Heterogeneous GNN | 6.327261500002351 | 0.085291600000346 | 0.0042645800000173 | 112866 | 18 |  |  |  |
