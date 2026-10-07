# Methods

## Study Data and Split

The complete image-quality cohort contained 1,785 images: 1,164 gradable images and 621 ungradable `NA` images. Patient-level splitting assigned 1,071 images to training, 357 to development validation, and 357 to the frozen internal test set. The respective gradable/NA counts were 699/372, 233/124, and 232/125.

The expanded gradable development manifest contained 1,471 records. Exclusion of 134 anomalous grouped records and removal of 383 duplicate registrations produced the final 954-image gradable cohort. Twenty-three opacity grade 1/2 boundary images underwent blinded single-clinician review; 13 labels changed. The 22 patient groups represented by reviewed images were assigned to training before the remaining groups were stratified. The fixed split contained 777 training images and 177 internal-validation images, with no patient-group or eye-group overlap.

The opacity class distribution was 149/290/225/113 for grades 0/1/2/3 in training and 34/66/51/26 in internal validation. The split seed was `877`.

NA images were used only for the gradable-versus-ungradable quality task. They were excluded from opacity-grade and opening-area losses because they did not carry reliable labels for those targets. At inference, an image classified as NA stops before grading and area estimation and is referred for review.

## Image Processing

Images were corrected for EXIF orientation and converted to RGB. Each image was resized without changing its aspect ratio and centered on a black `320 x 320` canvas. The upper-left 35% width by 6% height region was masked to suppress burned-in information. Pixel values were normalized using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

## Model

A ConvNeXt-Tiny encoder initialized from TorchVision ImageNet-1K weights and transferred through the earlier CCS task produced frozen image features. Opacity features concatenated global average, global RMS, and center-region average pools into a 2,304-dimensional vector. The four-class head used `LayerNorm(2304)`, dropout 0.55, a 128-unit linear layer, GELU, dropout 0.55, and a four-unit output layer.

The opening-area head concatenated the 768-dimensional global feature with normalized WTW (`WTW/11.3`) and used `LayerNorm(769)`, a 192-unit linear layer, GELU, dropout 0.50, and a scalar output. Sigmoid output represented an area fraction, converted by `area_mm2 = fraction * WTW_mm^2`.

## Optimization

The initial opacity head used AdamW with learning rate `1e-4`, weight decay 0.10, batch size 32, cosine annealing to `5e-6`, and early stopping patience 25 over at most 200 epochs. The loss combined class-weighted cross entropy with label smoothing 0.08 and a 0.20-weighted ordinal Smooth L1 term. Training-only regularization used Gaussian feature noise with standard deviation 0.025 and feature MixUp with probability 0.50 and mixing range 0.65-0.85.

Reviewed-label fine-tuning started from each seed's initial head and used AdamW at `2e-5`, cosine annealing to `1e-6`, weight decay 0.10, at most 120 epochs, and patience 20. Label smoothing was 0.06, ordinal loss weight 0.15, feature-noise standard deviation 0.015, and MixUp probability 0.35 with mixing range 0.70-0.90. Epoch-zero rollback protection retained the original head unless validation selection score improved.

The area head used AdamW at `5e-5`, cosine annealing to `2.5e-6`, weight decay 0.10, batch size 32, at most 250 epochs, and patience 30. Its objective combined three times the WTW-fraction Smooth L1 loss with a 0.10-weighted boundary loss at 16 and 20 mm2. Gaussian feature noise with standard deviation 0.01 was applied to training features.

## Derived Outcomes

Predicted opening area defined contraction as class 0 for area greater than 20 mm2, class 1 for area greater than 16 and at most 20 mm2, and class 2/3 for area at most 16 mm2. Final CCS diagnosis was negative unless opacity was grade 3 with area at most 20 mm2; grade-3 cases with area greater than 16 and at most 20 mm2 were mild positive, and grade-3 cases with area at most 16 mm2 were severe positive.

## Evaluation

Five independent head-training runs used seeds `20260927`, `20260928`, `20260929`, `20260930`, and `20261001` on the same fixed split. Opacity evaluation included accuracy, balanced accuracy, Macro-F1, macro one-vs-rest ROC-AUC, and grade-3 sensitivity and specificity. Area evaluation included MAE, RMSE, signed bias, and the fractions within 1 and 2 mm2. Contraction and final CCS evaluation included accuracy, balanced accuracy, Macro-F1, and sensitivity and specificity for the key severe category. Results are reported as the mean and sample standard deviation across seeds.

## Limitations

This development release uses internal validation only; the locked independent test set was not accessed. Reviewed boundary cases were selected from prior model errors and assigned to training, so the fixed split differs from the earlier development split. Rare contraction and CCS-positive categories have small validation support. Across-seed standard deviations quantify optimization stability rather than patient-level 95% confidence intervals. Exact framework, CUDA, and cuDNN patch versions were not preserved in the result artifact.
