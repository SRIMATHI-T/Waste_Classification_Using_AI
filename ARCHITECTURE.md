```text
                         WASTE CLASSIFICATION AI
                                  │
                                  ▼
                         TrashNet Dataset
                                  │
                                  ▼
                         Dataset Verification
                                  │
                                  ▼
                       EDA / Class Distribution
                                  │
                                  ▼
                     Preprocessing & Normalization
                                  │
                                  ▼
                      Stratified 70 / 15 / 15 Split
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │     BASELINE        │
                       │                     │
                       │ CNN From Scratch    │
                       │ Transfer Learning   │
                       └──────────┬──────────┘
                                  │
                                  ▼
                          ResNet50 Baseline
                                  │
                                  ▼
                        Fine-Tuning Stage A
                     Frozen Backbone + FC
                                  │
                                  ▼
                        Fine-Tuning Stage B
                       Unfreeze Layer4 + FC
                                  │
                                  ▼
                       Baseline Evaluation
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ IMPROVED TRAINING        │
                    │                          │
                    │ Strong Augmentation      │
                    │ AdamW                    │
                    │ Class Weighting          │
                    │ Discriminative LR        │
                    │ Cosine LR Scheduler      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                         Improved ResNet50
                                 │
                                 ▼
                         Add CBAM Attention
                                 │
                                 ▼
                        ResNet50 + CBAM
                                 │
                                 ▼
                    Deeper Fine-Tuning / Exp.5
                                 │
                                 ▼
                         Final Evaluation
                                 │
              ┌──────────────────┼─────────────────┐
              ▼                  ▼                 ▼
          Accuracy          Macro F1        Confusion Matrix
              │                  │                 │
              └──────────────────┼─────────────────┘
                                 ▼
                            Grad-CAM
                                 │
                                 ▼
                       Explain Model Attention
                                 │
                                 ▼
                       Save Trained Model
                                 │
                                 ▼
                       Streamlit Web UI
                                 │
                                 ▼
                       Upload Waste Image
                                 │
                                 ▼
                       ResNet50 + CBAM
                                 │
                                 ▼
                       Predicted Category
                                 │
                                 ▼
                    Confidence + Probabilities
```
