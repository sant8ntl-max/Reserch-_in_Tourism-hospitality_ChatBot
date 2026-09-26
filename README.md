# Tourism & Hospitality Conversational AI Bot

**Capstone Project — Industry-Specific Conversational AI**
**Author:** Santosh Kumar

## Overview
This project fine-tunes a pretrained Hugging Face language model (DistilGPT2) on a
custom Q&A dataset for the **Tourism & Hospitality** industry, covering hotel booking
queries, cancellation policies, meal plans, and guest services — enriched with
data-driven Q&A pairs generated from a real tourism destinations dataset. It also
includes supporting exploratory data analysis on two real datasets and a simple
content-based destination recommender.

## Repository Structure
```
├── Tourism_Hospitality_Conversational_Bot_Santosh_Kumar.ipynb   # Main Colab notebook (bot pipeline + EDA + recommender)
├── SK_HOTEL_BOOKING_ANALYSIS_EDA_Santosh_Kumar_CORRECTED.ipynb  # Supporting hotel-booking EDA notebook
├── tourism_hospitality_qa.csv                                   # Custom Q&A dataset (72 pairs)
├── Hotel_Bookings.csv                                            # Hotel booking dataset (119,390 records)
├── tourism_with_id.csv                                           # Indonesia Tourism Destination: places (Kaggle)
├── user.csv                                                      # Indonesia Tourism Destination: users (Kaggle)
├── tourism_rating.csv                                            # Indonesia Tourism Destination: ratings (Kaggle)
├── package_tourism.csv                                           # Indonesia Tourism Destination: packages (Kaggle)
├── requirements.txt                                              # Python dependencies
└── README.md
```

## Pipeline
1. **Dataset** — 72 Q&A pairs: 59 hand-crafted (booking, cancellation, meal plans,
   deposit types, customer types, guest services) + 13 generated directly from the
   real tourism destination dataset (categories, cities, pricing).
2. **Preprocessing** — Formatted as `Q: ... \n A: ...` for causal language modeling,
   train/validation split (85/15).
3. **Model** — `distilgpt2` from Hugging Face, fine-tuned using the `Trainer` API.
4. **Training** — 25 epochs on Google Colab (GPU or CPU), evaluation loss logged
   every epoch, checkpoints saved to Google Drive for persistence.
5. **Evaluation** — Training/validation loss curves, perplexity score (final:
   validation loss 3.4259, perplexity 30.75).
6. **Testing** — Sample conversations tested against the fine-tuned model, plus a
   live Gradio chat demo.
7. **Bonus — Tourism Destination Analysis** — EDA (category/city distribution, price
   vs. rating, average rating by category) and a TF-IDF content-based recommender
   on the Indonesia Tourism Destination dataset (Kaggle: `aprabowo/indonesia-tourism-destination`).

## How to Run
1. Open `Tourism_Hospitality_Conversational_Bot_Santosh_Kumar.ipynb` in Google Colab.
2. GPU is optional — the dataset is small enough to fine-tune on CPU if no GPU quota
   is available. If you have GPU access: `Runtime → Change runtime type → T4 GPU`.
3. Run the install cell, then **Runtime → Restart session**, then run the rest of the
   notebook top to bottom.
4. Upload the 6 CSV files when prompted: `tourism_hospitality_qa.csv`, `Hotel_Bookings.csv`,
   `tourism_with_id.csv`, `user.csv`, `tourism_rating.csv`, `package_tourism.csv`.
   These are saved to Google Drive automatically so you won't need to re-upload them
   on future runs.
5. The fine-tuned model is saved to `./tourism_bot_final` (and to Drive) and can be
   downloaded as a zip.
6. Run the final Gradio cell **separately** (not part of "Run all") to launch the
   live chat demo.

## Results

| Metric                  | Value   |
|--------------------------|---------|
| Final Training Loss       | ~0.20   |
| Final Validation Loss     | 3.4259  |
| Perplexity                 | 30.75   |

Training loss decreased steadily across all 25 epochs, while validation loss rose
after an early dip — a sign of overfitting, expected given the small (72-pair)
dataset. Manual review of 5 sample bot responses found 2 fully coherent, 2 topically
correct but repetitive, and 1 failed (empty) response, consistent with this
overfitting pattern.

Supporting EDA on the Hotel Bookings dataset found that cancellation rate rises
sharply with booking lead time — from 10.98% for bookings made within a week of
arrival to 67.66% for bookings made 365+ days in advance — making lead time the
single strongest predictor of cancellation in this dataset.

## Dataset Sources
- `Hotel_Bookings.csv` — publicly available hotel booking demand dataset (119,390 records).
- Indonesia Tourism Destination dataset — Prabowo, A. (2023). Kaggle:
  https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination

## Notes
This repository was built as part of an academic capstone project. The Q&A dataset,
model choice, and analysis write-up reflect the author's own work; the notebook code
follows standard Hugging Face `transformers` fine-tuning practices.
