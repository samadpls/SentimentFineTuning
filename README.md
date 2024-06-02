
# Fine-Tuning LLM for Sentiment Analysis
<img src='https://github.com/samadpls/SentimentFineTuning/assets/94792103/804a99db-becc-41d1-a356-64d292dc1985' align=right height=200px>

This repository includes the steps I took to fine-tune a large language model (LLM) for sentiment analysis using the IMDB dataset. The process leverages Hugging Face's `transformers` and `datasets` libraries, as well as the `peft` library for parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation).

The fine-tuned sentiment analysis model can be found on Hugging Face at: [samadpls/sentiment-analysis](https://huggingface.co/samadpls/sentiment-analysis).

## Model

We use the `distilbert-base-uncased` model as the base model for fine-tuning. The model is configured for sequence classification with two labels (positive and negative sentiment).

## LoRA Configuration

LoRA (Low-Rank Adaptation) is used to fine-tune the model efficiently by adding trainable low-rank adaptation matrices to certain model layers. The configuration parameters for LoRA used in this project are:

By leveraging LoRA, we achieve efficient fine-tuning with reduced computational resources, making it a key component of this project.

## Evaluation

The model's performance is evaluated using accuracy as the metric. The trained model is tested on a set of example texts to verify its predictions.


## Conclusion
 I developed this project in 2023 while learning to run and fine-tune LLMs. This project serves as a starting point for fine-tuning LLMs. Using LoRA, we can efficiently fine-tune large models with reduced computational resources. Feel free to experiment with various models, datasets, and configurations to enhance your understanding and achieve better results.
