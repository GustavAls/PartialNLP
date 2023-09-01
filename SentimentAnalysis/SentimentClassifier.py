import os
import torch
from datasets import load_dataset
import evaluate
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer)
import numpy as np
# Heavily based on: https://huggingface.co/blog/sentiment-analysis-python, and
# https://huggingface.co/docs/transformers/tasks/sequence_classification


class SentimentClassifier:
    def __init__(self, network_name, id2label, label2id):
        self.tokenizer = AutoTokenizer.from_pretrained(network_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(network_name, num_labels=2, id2label=id2label, label2id=label2id)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    @staticmethod
    def load_text_dataset(dataset_name="imdb"):
        data = load_dataset(dataset_name)
        train_data = data["train"].shuffle(seed=42).select([i for i in list(range(3000))])
        test_data = data["test"].shuffle(seed=42).select([i for i in list(range(300))])
        del data
        return train_data, test_data

    def tokenize(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    @staticmethod
    def compute_metrics(eval_pred):
        load_accuracy = evaluate.load("accuracy")
        load_f1 = evaluate.load("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return accuracy
        # return {"accuracy": accuracy, "f1": f1}

    def runner(self, train=True):
        train_data, test_data = self.load_text_dataset(dataset_name="imdb")
        tokenized_train = train_data.map(self.tokenize, batched=True, batch_size=100)
        tokenized_test = test_data.map(self.tokenize, batched=True, batch_size=100)

        training_args = TrainingArguments(output_dir="Test",
                                          learning_rate=2e-5,
                                          per_device_train_batch_size=4,
                                          per_device_eval_batch_size=4,
                                          num_train_epochs=2,
                                          weight_decay=0.01,
                                          save_strategy="epoch",
                                          evaluation_strategy="epoch")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self.tokenize,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics
        )
        if train:
            trainer.train()
            print("Training is done")
        else:
            trainer.evaluate()
            print("Evaluation is done")


if __name__ == "__main__":
    torch.cuda.is_available()
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    sentiment_classifier = SentimentClassifier("distilbert-base-uncased", id2label=id2label, label2id=label2id)
    sentiment_classifier.runner(train=True)


