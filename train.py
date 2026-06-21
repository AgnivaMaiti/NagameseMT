import os
import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed
)

def main():
    # 1. Reproducibility
    set_seed(42)
    
    # 2. Configuration
    model_checkpoint = "facebook/nllb-200-distilled-600M"
    dataset_name = "agnivamaiti/nagamese-english-mt"
    
    # Surrogate Token Strategy: 
    # Source is English (eng_Latn), Target is mapped to Assamese Bengali script (asm_Beng) during training.
    src_lang = "eng_Latn"
    tgt_lang = "asm_Beng"
    
    # 3. Load Tokenizer & Model
    tokenizer = NllbTokenizer.from_pretrained(
        model_checkpoint, 
        src_lang=src_lang, 
        tgt_lang=tgt_lang
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    # 4. Load & Preprocess Dataset
    raw_datasets = load_dataset(dataset_name)
    
    max_length = 128
    
    def preprocess_function(examples):
        inputs = [ex["en"] for ex in examples["translation"]]
        targets = [ex["nag"] for ex in examples["translation"]]
        
        model_inputs = tokenizer(
            inputs, max_length=max_length, truncation=True
        )
        
        # Tokenize targets with the target language forced to asm_Beng
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_length, truncation=True
            )
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    
    # 5. Metrics (SacreBLEU for validation)
    metric = evaluate.load("sacrebleu")
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}
        
    # 6. Training Arguments
    args = Seq2SeqTrainingArguments(
        output_dir="nllb-200-en-nagamese",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=15,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        lr_scheduler_type="linear",
        warmup_steps=500,
        generation_max_length=max_length,
        generation_num_beams=5,
        # Matching paper generation constraints
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        seed=42
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # 7. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 8. Train
    print("Starting full fine-tuning...")
    trainer.train()
    
    # 9. Save Best Model
    trainer.save_model("nllb-200-en-nagamese-best")
    print("Training complete. Best model saved to 'nllb-200-en-nagamese-best'")

if __name__ == "__main__":
    main()
