import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          DataCollatorForSeq2Seq,
                          AutoTokenizer)


class TextSummarizer:
    def __init__(self, checkpoint: str = "t5-small", prefix: str = "summarize: "):
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.prefix = prefix
        self.rouge = evaluate.load("rouge")
        self.collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.checkpoint)

    def preprocess_function(self, examples):
        inputs = [self.prefix + doc for doc in examples["text"]]
        model_inputs = self.tokenizer(inputs, max_length=1024, truncation=True)

        labels = self.tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def runner(self, train: bool=True):
        billsum = load_dataset("billsum", split="ca_test")
        billsum = billsum.train_test_split(test_size=0.2)

        tokenized_billsum = billsum.map(self.preprocess_function, batched=True)

        model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)

        training_args = Seq2SeqTrainingArguments(
            output_dir="my_awesome_billsum_model",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            predict_with_generate=True,
            fp16=True
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_billsum["train"],
            eval_dataset=tokenized_billsum["test"],
            tokenizer=self.tokenizer,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics,
        )

        if train:
            trainer.train()
            print("Training done")
        else:
            trainer.evaluate()


if __name__ == "__main__":
    torch.cuda.is_available()
    text_summarizer = TextSummarizer(checkpoint="t5-small", prefix="summarizer: ")
    text_summarizer.runner(train=True)
