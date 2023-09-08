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
import argparse
# Heavily based on: https://huggingface.co/blog/sentiment-analysis-python, and
# https://huggingface.co/docs/transformers/tasks/sequence_classification


class SentimentClassifier:
    def __init__(self, network_name, id2label, label2id, train_size=300, test_size=30):
        self.tokenizer = AutoTokenizer.from_pretrained(network_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(network_name,
                                                                        num_labels=2,
                                                                        id2label=id2label,
                                                                        label2id=label2id)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.train_size = train_size
        self.test_size = test_size

    def load_text_dataset(self, dataset_name="imdb"):
        data = load_dataset(dataset_name)
        train_data = data["train"].shuffle(seed=42).select([i for i in list(range(self.train_size))])
        test_data = data["test"].shuffle(seed=42).select([i for i in list(range(self.test_size))])
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
        return {"accuracy": accuracy, "f1": f1}

    def runner(self, output_path, train_bs, eval_bs, num_epochs, dataset_name, train=True):
        train_data, test_data = self.load_text_dataset(dataset_name=dataset_name)
        tokenized_train = train_data.map(self.tokenize, batched=True, batch_size=train_bs)
        tokenized_test = test_data.map(self.tokenize, batched=True, batch_size=eval_bs)

        training_args = TrainingArguments(output_dir=output_path,
                                          learning_rate=2e-5,
                                          per_device_train_batch_size=8,
                                          per_device_eval_batch_size=8,
                                          num_train_epochs=num_epochs,
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
    parser = argparse.ArgumentParser(
        description="Run training and or evaluation of Sentiment Classifier"
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dataset_name", type=str, default="imdb")
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--train_size", type=int, default=3000)
    parser.add_argument("--test_size", type=int, default=300)

    args = parser.parse_args()
    torch.cuda.is_available()
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    sentiment_classifier = SentimentClassifier("distilbert-base-uncased",
                                               id2label=id2label,
                                               label2id=label2id,
                                               train_size=args.train_size,
                                               test_size=args.test_size)
    sentiment_classifier.runner(args.output_path,
                                args.train_batch_size,
                                args.eval_batch_size,
                                args.num_epochs,
                                args.dataset_name,
                                train=args.train)

