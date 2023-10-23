import os

import evaluate
import numpy as np
import torch
import argparse
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          DataCollatorForSeq2Seq,
                          AutoTokenizer)

# Based on: https://huggingface.co/docs/transformers/tasks/summarization


class TextSummarizer:
    def __init__(self, checkpoint: str = "t5-small", prefix: str = "summarize: "):
        self.checkpoint = checkpoint
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.prefix = prefix
        self.rouge = evaluate.load("rouge")
        self.collator = DataCollatorForSeq2Seq(tokenizer=self._tokenizer, model=self.checkpoint)

    def preprocess_function(self, examples):
        inputs = [self.prefix + doc for doc in examples["text"]]
        model_inputs = self._tokenizer(inputs, max_length=1024, truncation=True)

        labels = self._tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self._tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self._tokenizer.pad_token_id)
        decoded_labels = self._tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self._tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def runner(self, output_path, num_epochs, dataset_name, device_batch_size, train: bool=True):
        billsum = load_dataset(dataset_name, split="ca_test")
        billsum = billsum.train_test_split(test_size=0.2)

        tokenized_billsum = billsum.map(self.preprocess_function, batched=True)

        model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_path,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=device_batch_size,
            per_device_eval_batch_size=device_batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=num_epochs,
            predict_with_generate=True,
            fp16=True
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_billsum["train"],
            eval_dataset=tokenized_billsum["test"],
            tokenizer=self._tokenizer,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics,
        )

        if train:
            trainer.train()
            print("Training done")
        else:
            trainer.evaluate()
            print("Evaluation done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training and or evaluation of Text summarizer Classifier"
    )

    parser.add_argument("--output_path", type=str, default=os.getcwd())
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dataset_name", type=str, default="billsum")
    parser.add_argument("--device_batch_size", type=int, default=2)
    parser.add_argument("--train", type=bool, default=True)

    args = parser.parse_args()
    torch.cuda.is_available()
    text_summarizer = TextSummarizer(checkpoint="t5-small", prefix="summarizer: ")

    text_summarizer.runner(output_path=args.output_path,
                           num_epochs=args.num_epochs,
                           dataset_name=args.dataset_name,
                           device_batch_size=args.device_batch_size,
                           train=args.train)


