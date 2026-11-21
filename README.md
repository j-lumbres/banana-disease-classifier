# Banana Disease Classifier

A MobileNetV2 image classifier for banana fruit and leaf diseases, served as a static web app that runs the model in the browser.

- Live demo: [Hugging Face Space](https://huggingface.co/spaces/j-lumbres/banana-disease-classifier)
- PyTorch weights: [Hugging Face model repo](https://huggingface.co/j-lumbres/banana-disease-mobilenet)

## Result

The deployed model reaches 89.5% accuracy on 474 held-out test images across 8 classes. The first version reached 73.9% on 9 classes. Almost all of that gain comes from fixing the labels in the training data, not from training longer.

| Model | Classes | Valid accuracy | Test accuracy |
|---|---|---|---|
| v1, frozen backbone, original labels | 9 | n/a | 73.9% (502 images) |
| v1, scored on the cleaned 8-class data | 8 | 87.4% | 90.9% |
| v2, fine-tuned on the cleaned data (deployed) | 8 | 89.7% | 89.5% |

Table 1. Accuracy of both versions. The cleaned validation set has 603 images and the cleaned test set has 474. The v1 row on cleaned data maps its 9 outputs onto the 8 new classes.

The deployed model was chosen on validation accuracy, where v2 leads by 2.3 points. On the test set v1 leads by 1.4 points, which is about 7 images. The two versions are therefore close, and fine-tuning the whole network gives no clear gain over training only the final layer.

## What limited the first version

Fine-tuning the whole network on the original labels gave 75.5% test accuracy, against 75.9% for the frozen v1 model on the same deduplicated images. More training did not help, so the errors were traced back to the data. Two problems explained most of them.

The first problem is conflicting labels. Every image was hashed (MD5 of the file bytes) to find identical files. After the class merge described below, 74 pictures still appear under two different labels. 69 of them are filed as both Black Sigatoka and Panama Disease, which caps the accuracy any model can reach on those two classes. All copies of these pictures were removed.

The second problem is two fruit classes that look the same. Black Pitting/Rust and Fungal Disease both show ripe fruit with brown spots, and v1 swapped them 63 times in 241 fruit test images. They were merged into one class, `Banana_spotted_fruit_disease`.

Exact duplicates were also kept only once, in the order train, valid, test. In the original split, 20 test images were byte-identical copies of training images.

## Per-class result of the deployed model

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Crown Rot (fruit) | 0.514 | 0.731 | 0.603 | 26 |
| Healthy (fruit) | 0.600 | 1.000 | 0.750 | 18 |
| Spotted fruit disease | 0.971 | 0.852 | 0.908 | 196 |
| Scab Moth (leaf) | 0.909 | 1.000 | 0.952 | 10 |
| Black Sigatoka (leaf) | 0.985 | 0.855 | 0.915 | 76 |
| Healthy (leaf) | 0.990 | 0.990 | 0.990 | 100 |
| Black Leaf Streak (leaf) | 1.000 | 0.969 | 0.984 | 32 |
| Panama Disease (leaf) | 0.556 | 0.938 | 0.698 | 16 |

Table 2. Test metrics per class for v2.

The three rare classes (Crown Rot, fruit Healthy, Panama Disease) have high recall but low precision. The model predicts them too often. The class-weighted loss gives these classes 1.8 to 2.9 times the average weight, which is the likely cause.

## Limitations

All fruit images show a single banana on a black background. A photo of a banana on a table or in a hand looks different from anything in training, so predictions on such photos are less reliable than the test numbers suggest. The merged spotted-fruit class also means the model cannot tell Black Pitting/Rust apart from other fungal spotting.

## Data

The images come from the gated Hugging Face dataset [Saon110/bd-crop-vegetable-plant-disease-dataset](https://huggingface.co/datasets/Saon110/bd-crop-vegetable-plant-disease-dataset). Notebook 01 streams up to 500 training, 100 validation, and 100 test images per banana class and resizes them to 224 by 224 pixels. The images are not included in this repository. Accept the dataset conditions on its page before running the notebook.

## Repository layout

- `01_frozen_baseline.ipynb` downloads the data, trains a small CNN baseline (59.2% test accuracy) and the frozen MobileNetV2 (v1), and deploys v1.
- `02_cleaned_finetune.ipynb` cleans the labels, fine-tunes MobileNetV2 in two stages (head only, then all layers), compares it with v1, and exports ONNX.
- `deploy/` holds the static Space. `index.html` runs `banana_mobilenet.onnx` with ONNX Runtime Web, and `class_names.json` lists the 8 classes.

Notebook 02 expects the v1 weights at `model_v1/banana_mobilenet.pth`. They are commit `baddb54` in the Hugging Face model repo history.

## Running

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then run notebook 01 for the data, and notebook 02 for the fine-tuned model. Notebook 02 picks Apple's MPS backend when it is available and falls back to CUDA or CPU. On an Apple A18 Pro it trains in about 15 minutes.

## License

The code (notebooks, `deploy/index.html`, and the other source files) is released under the [MIT License](LICENSE).

The trained model is not covered by the MIT License. `deploy/banana_mobilenet.onnx` and the PyTorch weights in the Hugging Face model repo were trained on [Saon110/bd-crop-vegetable-plant-disease-dataset](https://huggingface.co/datasets/Saon110/bd-crop-vegetable-plant-disease-dataset), which is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The model files are therefore shared under the same license. They may be reused with credit to the dataset, for non-commercial purposes only, and derivatives must keep the same license.
