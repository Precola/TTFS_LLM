from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification

# model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-large-uncased")

model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-large-uncased",
    num_labels=2   #
)
print("HF")
# i = 0
# print(model)
# for name, params in model.named_parameters():
#     i = i + 1
#     print(i, name, params.size())

from datasets import load_dataset

dataset = load_dataset("glue", "sst2")  #


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

from transformers import TrainingArguments, Trainer
import evaluate

metric = evaluate.load("glue", "sst2")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
