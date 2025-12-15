# BiasWipe: Mitigating Unintended Bias in Text Classifiers through Model Interpretability

This repository contains the code for reproducing the EMNLP 2024 paper:  
**BiasWipe: Mitigating Unintended Bias in Text Classifiers through Model Interpretability**

📄 [Original Paper Link](https://aclanthology.org/2024.emnlp-main.1172/)  
**Authors:** Mamta, Rishikant Chigrupaatii, Asif Ekbal


**Reproduced By:** Ruth-Ann Armstrong, Faith Baca, Lydia Ignatova

# Environment
To use this repository, first set up a Python environment. We recommend using the [Anaconda](https://www.anaconda.com/docs/getting-started/miniconda/install) package management system. Create and activate a new environment with Python 3.11, then install the required dependencies:

```bash
conda create -n biaswipe python=3.11
conda activate biaswipe
pip install -r requirements.txt 
```

# Data
## Data Download 

### English Toxic Content Classification Dataset
The original BiasWipe paper fine-tunes models on the Wikipedia Talk Pages Toxicity dataset, which contains human-annotated labels for toxic and non-toxic comments. It can be downloaded here: [Wikipedia Talk Pages Toxicity](https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Toxicity/4563973). 

### Spanish Toxic Content Classification Dataset
To extend BiasWipe to a non-English setting, we additionally evaluate the method on the Clandestino dataset, which consists of Spanish-language toxic and non-toxic text. It can be downloaded here: [Clandestino](https://github.com/microsoft/Clandestino)

### Bias Evaluation Dataset
To evaluate unintended bias in classification performance, we follow the original paper and use ConversationAI’s English sentence template dataset, which contains short phrases with identity terms and toxicity labels. It can be downloaded here: [ConversationAI Sentence Templates](https://github.com/conversationai/unintended-ml-bias-analysis/tree/main/sentence_templates)


## Data Preprocessing

### Wikipedia Talk Pages Toxicity
The Wikipedia Talk Pages dataset has separate files corresponding to the comments and their corresponding toxicity annotations. In order to combine them into a usable dataset, run the following python script:

```bash
python data/preprocess_wikitalks.py \
  --comments_file /path/to/toxicity_annotated_comments.tsv \
  --annotations_file /path/to/toxicity_annotations.tsv \
  --output_dir /path/to/output_folder
```

This will generate `wikitalks_dataset.tsv` in the provided output directory. 

### Clandestino
The Clandestino dataset is provided as a JSON file. In order to convert it to the proper .tsv format, run the following bash script:

```bash
./scripts/json_to_cleaned_tsv.sh \
    /input/path/to/clandestino_no_ontology_42023.json \
    /save/path/for/clandestino.tsv
```


### Train / Dev / Test Split
Before training models on any of the toxic classification datasets, we split the data into train, development (dev), and test sets. The `data/split_dataset.py` python script loads a dataset and splits it into three files while preserving the proportion of toxic vs non-toxic examples in each.

It can be run with the `scripts/split_dataset.sh` bash script, which makes it easy to edit the command-line arguments:

* `--input`: Path to the input dataset (TSV or CSV) with an is_toxic column
* `--train_out`: Path to the `train.tsv` file (including `.tsv`)
* `--dev_out`: Path to the `dev.tsv` file (including `.tsv`) 
* `--test_out`: Path to the `test.tsv` file (including `.tsv`) 
* `--train_size`: Fraction of data to use for the training set (default: 0.6)
* `--dev_size`: Fraction of data to use for the development set (default: 0.2)
* `--test_size`: Fraction of data to use for the test set (default: 0.2)

in order to generate separate .tsvs for the stuff. 

### Bias Evaluation Dataset
In order to use the sentence templates to assess toxicity, we preprocess them with the following command:

```bash
python data/preprocess_templates.py \
  --input_path path/to/template.csv \
  --output_path path/to/processed.csv \
  --identity_terms_path ./data/identity_terms_en.txt
```

This preprocessing normalizes the text, ensures consistent formatting for column names, and extracts identity terms so the dataset can be used for bias evaluation later on.



# Model Fine-Tuning
The LLMs had to be fine-tuned for toxic comment classification. Following from the provided BiasWipe repository, we used PyTorch's pre-trained BERT and RoBERTa models as a starting point. 

There are 4 models which we fine-tuned, unbiased, and evaluated in this reproduction:

|          | BERT                        | RoBERTa                        |
|----------|----------------------------|--------------------------------|
| English  | bert-base-uncased           | roberta-base                   |
| Spanish  | dccuchile/bert-base-spanish-wwm-cased | bertin-project/bertin-roberta-base-spanish |


The `models/train_classifier.py` python script loads a dataset, loads a pretrained PyTorch model, fine-tunes the model on the dataset, and saves the resulting weights.

It can be run with the `scripts/train_classifier.sh` bash script, which makes it easy to edit the command-line arguments:

* `--train_tsv`: Path to your `train.tsv` dataset
* `--dev_tsv`: Path to your `dev.tsv` dataset
* `--test_tsv`: Path to your `test.tsv` dataset
* `--bert_model`: Pretrained model type (view table above)
* `--output_dir`: Directory where the fine-tuned model, predictions, and evaluation results are saved
* `--task_name`: Specifies the dataset processor (senti for binary classification)
* `--N`: Number of transformer layers/encoder blocks to use
* `--train_batch_size` / `--eval_batch_size`: Number of samples per batch during training and evaluation
* `--max_seq_length`: Maximum token length for each input; longer sequences are truncated, shorter sequences padded
* `--num_train_epochs`: Total number of passes over the training dataset
* `--learning_rate`: Optimizer learning rate
* `--do_train` / `--do_eval`: Flags to perform training and evaluation (include both)

# Model UnBiasing
The unbiased models can be used for yeah 

The `models/unlearn_entity.py` python script allows you to unbias a fine-tuned BERT or RoBERTa model for toxic classification. The method calculates Shapley values to identify the model weights that are the most associated with biased predictions, and then zeroes out the top weights contributing to bias, producing an "unbiased" model. 

It can be run with the `scripts/unlearn_entity.sh` bash script, which makes it easy to edit the command-line arguments:

* `--bert_model`: Which pre-trained model to use (e.g., bert-base-uncased)
* `--fine_tuned_model_file`: Path to the fine-tuned model checkpoint (`.bin`)
* `--forget_file`: Path to `.tsv` from ConversationAI Sentence Templates with examples containing the biased term 
* `--forget_neu_file`: Path to `.tsv` from ConversationAI Sentence Templates with examples containing the biased term which will be stripped of the biased term at run time
* `--entity_term`: The target word to unbias
* `--num_weights`: Number of model weights to zero out
* `--num_repetitions`: How many times to repeat Shapley value computation for stability
* `--log_file`: Path to `.txt` to save Shapley values to (optional)

# Evaluation

## Evaluating Accuracy

The `evals/evaluate_accuracy.py` python script calculates overall accuracy, FPR, and FNR for a fine-tuned model on a labeled dataset.

It can be run with the `scripts/eval_accuracy.sh` script, which makes it easy to edit the command-line arguments:

* `--model_file`: Path to the fine-tuned model checkpoint (`.bin`)
* `--bert_model`: Pre-trained model type (e.g., bert-base-uncased, roberta-base)
* `--eval_file`: CSV/TSV file with columns `comment` and `is_toxic`
* `--batch_size`: Batch size for evaluation (default: 16)
* `--max_seq_length`: Maximum sequence length for tokenizer (default: 128)
* `--output_file`: File to save the evaluation metrics


## Evaluating Bias
The `evals/eval_bias.py` python script calculates and writes out per-entity and overall bias metrics for a fine-tuned model, including:
* Accuracy per entity term
* False Positive Rate (FPR) per entity term
* False Negative Rate (FNR) per entity term
* Weighted overall FPR/FNR
* False Positive Equality Difference (FPED)
* False Negative Equality Difference (FNED)


It can be run with the `scripts/eval_bias.sh` bash script, which makes it easy to edit the command-line arguments:

* `--model_file`: Path to the fine-tuned model checkpoint (`.bin`)
* `--bert_model`: Pre-trained model type (e.g., bert-base-uncased, roberta-base)
* `--eval_file`: Path to the `.csv` with columns `comment`, `keyword` and `is_toxic`, from the Bias Evaluation Dataset (ConversationAI Sentence Templates)
* `--entity_terms_file`: Path to the `.txt` file with one entity term per line to evaluate on
* `--batch_size`: Batch size for evaluation (default: 16)
* `--max_seq_length`: Maximum sequence length for tokenizer (default: 128)
* `--output_file`: File to save the evaluation results


# Pretrained Models
A couple of fine-tuned and un-biased models are available in this [OneDrive folder](https://uscedu-my.sharepoint.com/:f:/r/personal/lignatov_usc_edu/Documents/BiasWipe%20Reproduction%20Model%20Weights?csf=1&web=1&e=9HIcQ9). 

The included model files are:

* `bert.bin`: bert-base-uncased fine-tuned for toxic content classification on the Wikipedia Talk Pages Toxicity dataset
* `bert_gay_100.bin`: `bert.bin` after unlearning "gay", pruning 100 weights
* `bert_gay_homosexual_100.bin`: `bert_gay_100.bin` after unlearning "homosexual", pruning 100 weights
* `bert_gay_homosexual_lesbian_100.bin`: `bert_gay_homosexual_100.bin` after unlearning "lesbian", pruning 100 weights. 


# Reproduction Results

We ran many configurations to generate the results presented below. Please see our project report (also availalbe in the [OneDrive folder](https://uscedu-my.sharepoint.com/:f:/r/personal/lignatov_usc_edu/Documents/BiasWipe%20Reproduction%20Model%20Weights?csf=1&web=1&e=9HIcQ9)) for interpretation and discussion of our findings. 


| Model    | Variant      | Type       | Accuracy (EN) | Accuracy (ES) | FPED (EN) | FPED (ES) | FNED (EN) | FNED (ES) |
|----------|-------------|------------|---------------|---------------|-----------|-----------|-----------|-----------|
| **BERT** | BERT        | WTP Eval   | 96.67         | 81.31         | -         | -         | -         | -         |
|          |             | WTP Test   | **96.78**     | 81.52         | -         | -         | -         | -         |
|          |             | Template   | 89.24         | **85.62**     | 10.34     | 20.78     | 6.53      | **4.07**  |
|          | Unlearn 1   | WTP Eval   | 96.31         | 67.71         | -         | -         | -         | -         |
|          |             | WTP Test   | 96.33         | 67.92         | -         | -         | -         | -         |
|          |             | Template   | 86.34         | 75.62         | 14.66     | 3.02      | **5.32**  | 21.37     |
|          | Unlearn 2   | WTP Eval   | 95.97         | 66.22         | -         | -         | -         | -         |
|          |             | WTP Test   | 95.93         | 67.35         | -         | -         | -         | -         |
|          |             | Template   | 82.49         | 72.52         | 12.89     | 2.01      | 11.40     | 19.46     |
|          | Unlearn 3   | WTP Eval   | 95.73         | 66.79         | -         | -         | -         | -         |
|          |             | WTP Test   | 95.79         | 69.05         | -         | -         | -         | -         |
|          |             | Template   | 79.39         | 72.08         | **11.86** | **0.87**  | 14.30     | 22.30     |
| **RoBERTa** | RoBERTa  | WTP Eval   | 96.34         | 78.95         | -         | -         | -         | -         |
|          |             | WTP Test   | **96.64**     | **79.41**     | -         | -         | -         | -         |
|          |             | Template   | 85.59         | 76.71         | 15.77     | 24.93     | 6.90      | 14.09     |
|          | Unlearn 1   | WTP Eval   | 96.29         | 63.04         | -         | -         | -         | -         |
|          |             | WTP Test   | 96.50         | 63.35         | -         | -         | -         | -         |
|          |             | Template   | 87.21         | 50.90         | 15.54     | 11.45     | **6.31**  | 5.70      |
|          | Unlearn 2   | WTP Eval   | 96.18         | 63.71         | -         | -         | -         | -         |
|          |             | WTP Test   | 96.32         | 63.19         | -         | -         | -         | -         |
|          |             | Template   | 88.95         | 50.10         | 6.22      | **0.83**  | 8.02      | **0.69**  |
|          | Unlearn 3   | WTP Eval   | 95.27         | 62.78         | -         | -         | -         | -         |
|          |             | WTP Test   | 95.45         | 62.83         | -         | -         | -         | -         |
|          |             | Template   | 71.95         | 50.06         | **4.64**  | 3.09      | 9.76      | 1.62      |


