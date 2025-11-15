# 📁 Recommended Project Structure for Hackathon

```
armeta_cv/
│
├── 📄 README.md                        # Main documentation
├── 📄 requirements.txt                 # Python dependencies
├── 📄 .gitignore                       # Git ignore rules
│
├── 📁 data/                            # Dataset
│   ├── pdf/                           # Original PDF documents (45 files)
│   ├── annotations/                   # JSON annotations
│   │   └── selected_annotations.json
│   ├── images/                        # Converted PNG (129 files)
│   ├── labels/                        # YOLO format labels (.txt)
│   ├── train/                         # Training split (80%)
│   │   ├── images/
│   │   └── labels/
│   ├── val/                           # Validation split (20%)
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml                      # YOLO dataset config
│
├── 📁 scripts/                         # Utility scripts
│   ├── prepare_data.py                # PDF → PNG conversion
│   ├── convert_annotations.py         # JSON → YOLO format
│   ├── split_dataset.py               # Train/val split
│   └── visualize_data.py              # Dataset visualization
│
├── 📁 training/                        # Training scripts
│   ├── train_yolov8s_optimized.py     # Main training script (YOLOv8s)
│   ├── train_yolov8n.py               # Baseline (YOLOv8n)
│   └── config.yaml                    # Training config
│
├── 📁 inference/                       # Inference scripts
│   ├── inference_optimized.py         # Batch inference
│   └── predict_single.py              # Single image prediction
│
├── 📁 runs/                            # Training outputs (auto-generated)
│   └── detect/
│       └── train*/
│           ├── weights/
│           │   ├── best.pt            # ⭐ Best model
│           │   └── last.pt
│           ├── results.csv            # Training metrics
│           └── *.png                  # Training plots
│
├── 📁 results/                         # Inference results
│   ├── predictions.json               # Detection results
│   └── visualizations/                # Annotated images
│
├── 📁 app/                             # Demo application
│   ├── streamlit_app.py               # Streamlit demo
│   └── utils.py                       # App utilities
│
├── 📁 notebooks/                       # Jupyter/Colab notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_colab.ipynb        # Your current Colab notebook
│   └── 03_results_analysis.ipynb
│
├── 📁 docs/                            # Documentation
│   ├── TRAINING.md                    # Training guide
│   ├── INFERENCE.md                   # Inference guide
│   └── METRICS.md                     # Results & metrics
│
└── 📁 presentation/                    # Hackathon materials
    ├── slides.pdf                     # Presentation (8-10 slides)
    ├── demo_video.mp4                 # Demo video (max 3 min)
    └── screenshots/                   # Screenshots for docs
```

## 🎯 Priority for Hackathon (Next 24h)

### Phase 1: Retrain with YOLOv8s (3-4h)
- [ ] Run `train_yolov8s_optimized.py` in Colab
- [ ] Monitor training (should finish in 30-60min)
- [ ] Compare metrics with YOLOv8n
- [ ] Download `best.pt`

### Phase 2: Inference & Analysis (2h)
- [ ] Run inference on full dataset (129 images)
- [ ] Export results to JSON
- [ ] Create visualizations
- [ ] Document improvements

### Phase 3: Demo App (4h)
- [ ] Create Streamlit app (see example below)
- [ ] Test with sample PDFs
- [ ] Record demo video (3 min max)

### Phase 4: Documentation (2h)
- [ ] README.md with setup instructions
- [ ] Results table (YOLOv8n vs YOLOv8s)
- [ ] GitHub repository cleanup

### Phase 5: Presentation (3h)
- [ ] 8-10 slides (Problem → Solution → Results → Impact)
- [ ] Practice demo
- [ ] Prepare for Q&A
