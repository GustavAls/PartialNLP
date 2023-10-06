import os
import torch
from datasets import load_dataset
import evaluate
from transformers import (AutoTokenizer,
                          DataCollatorWithPadding,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer)
from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification
import numpy as np
import argparse
# Heavily based on: https://huggingface.co/blog/sentiment-analysis-python, and
# https://huggingface.co/docs/transformers/tasks/sequence_classification


class SentimentClassifier:
    def __init__(self, network_name, id2label, label2id, train_size=300, test_size=30):
        self._tokenizer = AutoTokenizer.from_pretrained(network_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(network_name,
                                                                        num_labels=2,
                                                                        id2label=id2label,
                                                                        label2id=label2id)
        self.model.save_pretrained(os.path.join(os.getcwd(), "model"))
        self.collator = DataCollatorWithPadding(tokenizer=self._tokenizer)
        self.train_size = train_size
        self.test_size = test_size

    def load_text_dataset(self, dataset_name="imdb"):
        data = load_dataset(dataset_name)
        train_data = data["train"].shuffle(seed=42).select([i for i in list(range(self.train_size))])
        test_data = data["test"].shuffle(seed=42).select([i for i in list(range(self.test_size))])
        del data
        return train_data, test_data

    def tokenize(self, examples):
        return self._tokenizer(examples["text"], return_tensors="pt", padding=True)

    @staticmethod
    def compute_metrics(eval_pred):
        load_accuracy = evaluate.load("accuracy")
        load_f1 = evaluate.load("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}

    def runner(self, output_path, train_bs, eval_bs, num_epochs, dataset_name, device_batch_size, train):
        train_data, test_data = self.load_text_dataset(dataset_name=dataset_name)
        tokenized_train = train_data.map(self.tokenize, batched=True, batch_size=train_bs)
        tokenized_test = test_data.map(self.tokenize, batched=True, batch_size=eval_bs)

        training_args = TrainingArguments(output_dir=output_path,
                                          learning_rate=2e-5,
                                          do_train=train,
                                          per_device_train_batch_size=device_batch_size,
                                          per_device_eval_batch_size=device_batch_size,
                                          num_train_epochs=num_epochs,
                                          evaluation_strategy="epoch",
                                          save_strategy="epoch",
                                          load_best_model_at_end=True,
                                          weight_decay=0.01)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self._tokenizer,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics
        )
        if train:
            trainer.train()
            print("Training is done")
        else:
            trainer.evaluate()
        print("Evaluation is done")


    def predict(self, text):
        tokenized_test_text = self._tokenizer(text, return_tensors="pt")
        tokenized_test_text = tokenized_test_text.to(self.model.device)
        with torch.no_grad():
            logits = self.model(**tokenized_test_text).logits
        predicted_class_id = logits.argmax().item()
        return self.model.config.id2label[predicted_class_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training and or evaluation of Sentiment Classifier"
    )
    parser.add_argument("--output_path", type=str, default=os.getcwd())
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dataset_name", type=str, default="imdb")
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--device_batch_size", type=int, default=4)

    args = parser.parse_args()
    torch.cuda.is_available()
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    sentiment_classifier = SentimentClassifier("distilbert-base-uncased",
                                               id2label=id2label,
                                               label2id=label2id,
                                               train_size=args.train_size,
                                               test_size=args.test_size)
    sentiment_classifier.runner(output_path=args.output_path,
                                train_bs=args.train_batch_size,
                                eval_bs=args.eval_batch_size,
                                num_epochs=args.num_epochs,
                                dataset_name=args.dataset_name,
                                device_batch_size=args.device_batch_size,
                                train=args.train)

