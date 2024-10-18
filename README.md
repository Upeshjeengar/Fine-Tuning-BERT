# Fine-Tuning-BERT-
**Fine-Tuning BERT for Phishing URL Identification**

BERT, short for Bidirectional Encoder Representations from Transformers, is a machine learning (ML) framework for natural language processing. In 2018, Google developed this algorithm to improve contextual understanding of unlabeled text across a broad range of tasks by learning to predict text that might come before and after (bi-directional) other text.


---


**USECASES**:
* Sentiment Analysis
* chatbot question answer
* Help predicts text when writing an email
* Can quickly summarize long legal contracts
* Differentiate words that have multiple meanings based on the surrounding text

---


## BERT vs GPT
**BERT**
* Bidirectional Can process text left-to-right and right- to-left. BERT uses the encoder segment of a transformation model.
* Applied in Google Docs, Gmail, smart compose, enhanced search, voice assistance, analyzing customer reviews, and so on.
* GLUE score = 80.4% and 93.3% accuracy on the SQUAD dataset.
* Uses two unsupervised tasks, masked language modeling, fill in the blanks and next sentence prediction e.g. does sentence B come after sentence A?

---

**GPT**
* Autoregressive and unidirectional. Text is processed in one direction. GPT uses the decoder segment of a transformation model.
* Applied in application building, generating ML code, websites, writing articles, podcasts, creating legal documents, and so on.
* 64.3% accuracy on the TriviaAQ benchmark and 76.2% accuracy on LAMBADA, with zero-shot learning
* Straightforward text generation using autoregressive language modeling

## Prerequisites:
1. If You don't have an account of Hugging Face, Create an account on [HF hub](https://huggingface.co/join). 
2. Create a new token [here](https://huggingface.co/settings/tokens).
3. Upload it into Secrets if using Google Colab or you can view [this](https://medium.com/@aroman11/how-to-use-hugging-face-api-token-in-python-for-ai-application-step-by-step-be0ed00d315c) article.

#### [Dataset](https://huggingface.co/datasets/shawhin/phishing-site-classification) I used for fine-tuning [BERT](https://huggingface.co/google-bert/bert-base-uncased)

## Brief summary

1. **Dataset Loading**:  
   You loaded the phishing site classification dataset from Hugging Face using `load_dataset`.

2. **Model Initialization**:  
   You defined the model path (`bert-base-uncased`) and set up a binary classification model using `AutoModelForSequenceClassification`, with two labels: "Safe" and "Not Safe". You froze all base model parameters except for the pooling layers to fine-tune only those.

3. **Preprocessing**:  
   You tokenized the dataset using a `preprocess_function`, which applies truncation to the text inputs. The tokenized dataset was created using `dataset_dict.map`.

4. **Data Collation**:  
   You used `DataCollatorWithPadding` to handle padding while batching the data for training and evaluation.

5. **Metrics Loading**:  
   You loaded accuracy and ROC AUC score metrics using the `evaluate` library.

6. **Metric Computation**:  
   In the `compute_metrics` function, you computed the ROC AUC and accuracy after applying softmax to the predictions to calculate class probabilities.

7. **Training Arguments**:  
   You defined hyperparameters including a learning rate of `2e-4`, batch size of 8, and trained for 10 epochs. You also set strategies for logging, evaluation, saving checkpoints, and loading the best model.

8. **Model Training**:  
   The `Trainer` class was initialized with the model, tokenizer, datasets, and metrics. The training process was executed using the `train()` method.

9. **Validation**:  
   After training, you predicted the validation set using the model and computed the accuracy and ROC AUC scores using the `compute_metrics` function.

10. **Saving the model** to drive
