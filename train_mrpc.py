import numpy as np
from datasets import load_dataset, load_metric
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer

checkpoint = "bert-base-uncased"
print("Starting ...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print(f"Loaded tokenizer {checkpoint}")
model: BertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
print(f"Loaded model {checkpoint}")

optimizer = AdamW(model.parameters())


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tokenized_datasets = load_dataset("glue", "mrpc").map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("test1", evaluation_strategy="epoch", save_strategy="epoch", push_to_hub=True)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

