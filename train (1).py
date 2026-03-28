import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import PeftModel

# base model
base_model_name = "meta-llama/Llama-3.2-1B"

# existing LoRA adapter
adapter_repo = "vitalune/llama-3.2-1b-kichwa"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# load existing LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    adapter_repo
)

# load dataset
dataset = load_dataset(
    "json",
    data_files="train.jsonl"
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,   # padding handled by collator
        max_length=2048
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
tokenized_dataset = tokenized_dataset["train"].train_test_split(
    test_size=0.1
)

# training arguments
training_args = TrainingArguments(
    output_dir="./new_adapter",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
    report_to="tensorboard"
)

# data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator
)

# train
trainer.train()

# save final adapter
trainer.save_model("./new_adapter")