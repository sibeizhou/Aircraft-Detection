# Aircraft Detection and Segmentation in Aerial Images

A computer vision project for detecting and segmenting aircraft in aerial images using PyTorch and Detectron2. The project fine-tunes Faster R-CNN for aircraft detection and trains a segmentation pipeline to generate aircraft masks for Kaggle-style evaluation.

## Results

| Task | Metric | Result |
|---|---:|---:|
| Aircraft segmentation | Mean IoU | 0.872 |
| Kaggle submission | Score | 0.775 |
| Kaggle leaderboard | Rank | 5th |

## Technical Highlights

- Fine-tuned a Detectron2 Faster R-CNN X-101-FPN model for single-class aircraft detection.
- Used transfer learning from COCO-pretrained weights.
- Configured object detection training with custom dataset registration and COCO-style evaluation.
- Trained a PyTorch segmentation model using `BCEWithLogitsLoss` to predict binary aircraft masks.
- Generated run-length encoded masks for Kaggle submission.
- Evaluated outputs using mean IoU and qualitative test-set visualizations.

## Model Pipeline

```text
Aerial Image
  -> Detectron2 Dataset Registration
  -> Faster R-CNN Aircraft Detection
  -> Aircraft Crop Extraction
  -> PyTorch Binary Mask Segmentation
  -> Mask Post-processing
  -> RLE Encoding
  -> Kaggle Submission
