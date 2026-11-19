# 4. Clock drawing test

**Task:** the person draws a clock (on screen, or uploads a photo of a paper drawing). Drawing a clock needs
planning, memory and spatial skill, so the shape of the drawing carries information.

**Output:** *flagged* or *not flagged*, plus an estimated clock score from 0 to 5.
Code: `app/clock.py`, `clock/` · Models: `models/clock_cropnet.onnx`, `models/clock_vit.onnx`, `models/clock_features.json`

## Method, step by step

1. **Clean the image** (`clock/preprocess.py`)
   - Flip it so the ink is bright on a dark background.
   - Remove solid filled boxes (the name boxes found on scanned forms).
   - Cut away empty margin and pad to a square 512×512.
   - A photo is first resized so its long side is 828 px, to match the scale of the training scans.
2. **Find and crop the clock — three times** (`clock/crop.py`, model `clock_cropnet.onnx`)
   A small network ("Crop-Net") produces a heat map of where the clock is. The image is cropped around the hot
   area, then the process repeats twice more, each time zooming in closer.
3. **ViT model** (`clock_vit.onnx`)
   The 224×224 crop goes into a Vision Transformer. It outputs two numbers: an estimated clock score (0–5) and the
   probability that the drawing looks like those from people with dementia.
4. **Decision.** If that probability is at or above a stored threshold (`clock_features.json`), the result is *flagged*.
   The wording shown is "Drawing resembles those of people with dementia in the training data" — never a diagnosis.

## How a Vision Transformer (ViT) works

Older image models slide small filters across a picture. A ViT works more like reading a sentence.

1. **Cut the picture into patches.** A 224×224 image becomes a grid of 16×16-pixel squares (196 of them). Each square is like a *word*.
2. **Turn each patch into numbers** (a list of 768 numbers) and add a *position tag*, so the model knows where in the drawing each patch came from.
3. **Attention.** Stacked layers let every patch look at every other patch and decide which matter for each other. This is how it can pick up "the numbers are crowded on one side" or "the hands don't meet at the centre" — relations between faraway parts.
4. **Head.** A small final network turns the combined understanding into the two outputs above.

We use **ViT-B/16**, already pre-trained on millions of everyday photos, then **fine-tuned** (re-trained a little) on
clock drawings. Starting from a pre-trained model is why a modest amount of drawing data is enough.

## Where the training data came from

NHATS (National Health and Aging Trends Study) clock drawings. This data is access-restricted, so it is **not** in
this project and was never uploaded to any AI platform or Kaggle. Training ran on Google Colab.

## Honest results and limits

- Measured on NHATS held-out people (1,089, grouped by person): balanced accuracy **0.654 (95% CI 0.605–0.707)**, AUC 0.707. It flags about a third of people while only about 5% in that data have dementia — so many "flags" are false alarms.
- The training images were scanned paper. **Photos of paper drawings and drawings made with a mouse are not validated** (a "domain shift").
- The cropping step did not learn very well.
- So a flag means "worth a closer look", nothing more.
