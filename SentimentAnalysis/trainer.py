import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def load_text_dataset(dataset_name="imdb"):
    imdb = load_dataset(dataset_name)
    train_data = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
    test_data = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])
    del imdb
    return train_data, test_data


def tokenize(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


def runner(train=True):
    train_data, test_data = load_text_dataset(dataset_name="imdb")
    tokenized_train = train_data.map(tokenize, batched=True)
    tokenized_test = test_data.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.fit()
    print("Training is done")
    trainer.evaluate()
    print("Evaluation is done")


if __name__ == "__main__":
    torch.cuda.is_available()
    runner(train=True)


