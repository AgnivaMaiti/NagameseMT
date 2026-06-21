import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

def main():
    # 1. Configuration
    # To run zero-shot baseline, set model_path = "facebook/nllb-200-distilled-600M"
    model_path = "./nllb-200-en-nagamese-best"
    dataset_name = "agnivamaiti/nagamese-english-mt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Load Tokenizer & Model
    print(f"Loading model from {model_path}...")
    tokenizer = NllbTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M", 
        src_lang="eng_Latn", 
        tgt_lang="asm_Latn" # The surrogate inference token
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    
    # Verify the surrogate token ID maps to <unk> (ID 3)
    surrogate_token_id = tokenizer.convert_tokens_to_ids("asm_Latn")
    print(f"Surrogate forced BOS token ID: {surrogate_token_id} (Expected: 3)")
    
    # 3. Load Test Data
    print(f"Loading {dataset_name} test split...")
    dataset = load_dataset(dataset_name, split="test")
    sources = [ex["en"] for ex in dataset["translation"]]
    references = [[ex["nag"]] for ex in dataset["translation"]]
    
    # 4. Evaluation Loop
    batch_size = 16
    predictions = []
    
    print("Starting generation...")
    for i in tqdm(range(0, len(sources), batch_size)):
        batch_src = sources[i : i + batch_size]
        
        inputs = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=surrogate_token_id,
                max_length=128,
                num_beams=5,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
            
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predictions.extend(decoded_preds)
        
    # 5. Compute Metrics
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    # COMET requires the Unbabel/wmt22-comet-da model
    comet = evaluate.load("comet", config_name="wmt22-comet-da")
    
    print("\nComputing metrics...")
    
    bleu_result = sacrebleu.compute(predictions=predictions, references=references)
    chrf_result = chrf.compute(predictions=predictions, references=references)
    
    # COMET requires a slightly different format (sources, predictions, references)
    comet_refs = [ref[0] for ref in references]
    comet_result = comet.compute(predictions=predictions, references=comet_refs, sources=sources)
    
    print("="*40)
    print("EVALUATION RESULTS (EN -> NAG)")
    print("="*40)
    print(f"SacreBLEU : {bleu_result['score']:.2f}")
    print(f"chrF      : {chrf_result['score']:.2f}")
    print(f"COMET     : {comet_result['mean_score']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
