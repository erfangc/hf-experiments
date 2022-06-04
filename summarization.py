import numpy as np
from datasets import load_dataset
from datasets import load_metric
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, T5TokenizerFast, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, \
    DataCollatorForSeq2Seq, Seq2SeqTrainer

rouge_score = load_metric("rouge")
english_dataset = load_dataset("amazon_reviews_multi", "en")
model_checkpoint = "google/mt5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(model_checkpoint)


#
# preprocessing function
#

def filter_books(example):
    return (
            example["product_category"] == "book"
            or example["product_category"] == "digital_ebook_purchase"
    )


max_input_length = 512
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"], max_length=max_input_length, truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["review_title"], max_length=max_target_length, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE score
    decoded_preds = ["\n".join(sent_tokenize(sent.strip())) for sent in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(sent.strip())) for sent in decoded_labels]

    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True,
    )

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


english_dataset = english_dataset.filter(filter_books)
tokenized_datasets = english_dataset.map(preprocess_function, batched=True)

batch_size = 8
num_train_epochs = 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-es",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

trainer.push_to_hub(commit_message="Training complete", tags="summarization")
