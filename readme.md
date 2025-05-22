# BiasWipe: Mitigating Unintended Bias in Text Classifiers through Model Interpretability

This repository contains the code for the EMNLP 2024 paper:  
**BiasWipe: Mitigating Unintended Bias in Text Classifiers through Model Interpretability**

📄 [Paper Link](https://aclanthology.org/2024.emnlp-main.1172/)  
**Authors:** Mamta, Rishikant Chigrupaatii, Asif Ekbal

## Instructions to run
```bash
#Step 0:
cd framework

# Step 1: Train the baseline model: Saves the trained model in the logs_original/ directory.

bash run.sh

# Step 2: Identify false positives (misclassified samples): Identifies misclassified samples (false positives) for bias analysis.

python misclassified.py

# Step 3: Debias the model for a specific entity: Saves the debiased(unlearnt) model in the weights_shap/ directory.

bash unlearn_single.sh
```

## Citation

If you use this code or find our work helpful, please cite the following paper:

```bibtex
@inproceedings{mamta-etal-2024-biaswipe,
    title = "{B}ias{W}ipe: Mitigating Unintended Bias in Text Classifiers through Model Interpretability",
    author = "Mamta, Mamta  and
      Chigrupaatii, Rishikant  and
      Ekbal, Asif",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1172/",
    doi = "10.18653/v1/2024.emnlp-main.1172",
    pages = "21059--21070",
    abstract = "Toxic content detection plays a vital role in addressing the misuse of social media platforms to harm people or groups due to their race, gender or ethnicity. However, due to the nature of the datasets, systems develop an unintended bias due to the over-generalization of the model to the training data. This compromises the fairness of the systems, which can impact certain groups due to their race, gender, etc.Existing methods mitigate bias using data augmentation, adversarial learning, etc., which require re-training and adding extra parameters to the model.In this work, we present a robust and generalizable technique \textit{BiasWipe} to mitigate unintended bias in language models. \textit{BiasWipe} utilizes model interpretability using Shapley values, which achieve fairness by pruning the neuron weights responsible for unintended bias. It first identifies the neuron weights responsible for unintended bias and then achieves fairness by pruning them without loss of original performance. It does not require re-training or adding extra parameters to the model. To show the effectiveness of our proposed technique for bias unlearning, we perform extensive experiments for Toxic content detection for BERT, RoBERTa, and GPT models. ."
}
