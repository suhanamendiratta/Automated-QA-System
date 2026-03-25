from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("ms_marco", "v1.1")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

def preprocess(example):
    return tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length"
    )

train_dataset = dataset["train"].map(preprocess)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()