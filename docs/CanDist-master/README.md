# Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation

This is a PyTorch implementation of our ACL 2025 paper CanDist.

## An example of running CanDist on *Banking*

**Step 1. Data Preparation**

The data should be placed under the `data` folder with the following structure:

```
data-in
 |-- banking  # data for banking dataset
 |    |-- train.csv # for transductive evaluation
 |    |-- test.csv  # for inductive evaluation
 |    |-- dev.csv   # for validation
 |    |-- ...
 |-- ...
```

**Step 2. Prompt Candidate Annotations by LLM**

Please execute the commands in `llm_annotation.ipynb` to generate candidate annotations.

**Step 3. Distill Candidate Annotations by SLM**

Before training the SLM, please conduct data augmentation using back-translation:

```shell
python backtrans.py 
```

Then, you can train the SLM model using the following command:

```shell
bash slm_run.sh 0 banking 77 train_add_bt
```