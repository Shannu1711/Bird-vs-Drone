# Bird vs Drone

[Open the live app](https://bird-vs-drone-g7curq3qyzyxahjjd9amuf.streamlit.app/) · [GitHub repository](https://github.com/Shannu1711/Bird-vs-Drone)

**Shanmukha Pranay · Individual project**

Upload a JPG or PNG containing a bird or a drone. The Streamlit app displays your photo, the predicted class, and the model’s confidence for that image.

## Results

MobileNetV2 correctly classified **213 of 215 held-out images**. Both models used the same dataset splits. The notebook selected the model with the higher validation macro F1 before evaluating the test set. Macro F1 gives equal weight to the bird and drone classes.

| Model | Validation macro F1 | Test accuracy | Test macro F1 | Training time |
|---|---:|---:|---:|---:|
| Custom CNN | 0.7094 | 80.47% | 0.7910 | 81.3 min |
| Fine-tuned MobileNetV2 | 0.9796 | 99.07% | 0.9905 | 51.0 min |

Training times are from this project’s Colab run. The notebook includes the per-class precision and recall, confusion matrices, and training curves.

## Dataset and training

The course dataset contains 3,319 labeled photographs.

| Split | Bird | Drone | Total |
|---|---:|---:|---:|
| Training | 1,414 | 1,248 | 2,662 |
| Validation | 217 | 225 | 442 |
| Test | 121 | 94 | 215 |

Images are resized to 224 × 224 pixels. Training uses random flips, rotations, crops, zoom, and brightness changes. The custom CNN learns from scratch. MobileNetV2 starts with ImageNet weights, trains a classifier head, then fine-tunes its upper layers. Validation loss guides checkpoint saving and early stopping.

## Run the app on Windows

With Python 3.11 installed, download and extract the repository. Open Command Prompt in the folder containing `app.py` and run these commands in order.

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal and upload a photo. The app loads the included trained model and requires no API key.

## Reproduce the experiment

Open `Bird_vs_Drone (1).ipynb` in Google Colab and select **Runtime → Run all**. At the upload prompt, choose the instructor’s original train, valid, and test ZIP files together. The last cell exports the trained models, evaluation tables, plots, and Streamlit app.

| File | Purpose |
|---|---|
| `app.py` | Image upload, preprocessing, prediction, and display |
| `best_model.keras` | Trained MobileNetV2 model used by the app |
| `Bird_vs_Drone (1).ipynb` | Executed training and evaluation notebook |
| `requirements.txt` | Python packages needed to run the app |

## Interpreting a prediction

The model assigns one label to the whole photograph. Its confidence score describes that single prediction. It can still give a high score for an unrelated image or a scene containing both classes. The reported accuracy measures performance on the supplied test split.
