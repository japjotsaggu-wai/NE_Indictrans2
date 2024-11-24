import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10
from IndicTransToolkit import IndicProcessor
from nltk import sent_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA
from peft import PeftModel
from mosestokenizer import MosesSentenceSplitter
from config import (
    base_ckpt_dir,
    lora_ckpt_dir,
    BATCH_SIZE,
    flores_codes,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def split_sentences(input_text, lang):
    if lang == "eng_Latn":
        input_sentences = sent_tokenize(input_text)
        with MosesSentenceSplitter(flores_codes[lang]) as splitter:
            sents_moses = splitter([input_text])
        sents_nltk = sent_tokenize(input_text)
        if len(sents_nltk) < len(sents_moses):
            input_sentences = sents_nltk
        else:
            input_sentences = sents_moses
        input_sentences = [sent.replace("\xad", "") for sent in input_sentences]
    else:
        input_sentences = sentence_split(
            input_text, lang=flores_codes[lang], delim_pat=DELIM_PAT_NO_DANDA
        )
    return input_sentences

def initialize_model_and_tokenizer(base_ckpt_dir, lora_ckpt_dir, quantization, attn_implementation):
    # Configure quantization
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_use_double_quant=True, bnb_8bit_compute_dtype=torch.bfloat16)
    else:
        qconfig = None

    # Verify and configure attention
    if attn_implementation == "flash_attention_2" and is_flash_attn_2_available() and is_flash_attn_greater_or_equal_2_10():
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "eager"

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(base_ckpt_dir, trust_remote_code=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_ckpt_dir,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )
    if qconfig is None:
        base_model = base_model.to(DEVICE).half()
    base_model.eval()

    # Load LoRA model
    lora_model = PeftModel.from_pretrained(base_model, lora_ckpt_dir)
    return tokenizer, lora_model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations


def main():
    path_to_csv = sys.argv[1] if len(sys.argv) > 1 
    quantization = sys.argv[2] if len(sys.argv) > 2 else ""
    attn_implementation = sys.argv[3] if len(sys.argv) > 3 else "eager"
    
    if not path_to_csv:
        print("Usage: python script.py <path_to_csv> [quantization] [attn_implementation]")
        sys.exit(1)

    input_csv = pd.read_csv(path_to_csv)
    #new column
    input_csv["translation"] = None 
    
    ip = IndicProcessor(inference=True)

    for src_lang in input_csv['lang'].unique():
        tokenizer, lora_model = initialize_model_and_tokenizer(
            base_ckpt_dir, lora_ckpt_dir, quantization, attn_implementation
        )
    
        tgt_lang = "eng_Latn"  
        
        #group by language first
        lang_group = input_csv[input_csv["lang"] == src_lang] 
        sentences = lang_group["source"].tolist() 
    
        translations = batch_translate(sents, src_lang, tgt_lang, lora_model, tokenizer, ip)
        input_csv.loc[lang_group.index, "translation"] = translations
        
        del tokenizer, lora_model
    
    #saved to the same path
    input_csv.to_csv(path_to_csv) 

if __name__ == "__main__":
    main()
