import os
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10
from IndicTransToolkit import IndicProcessor
from nltk import sent_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA
from peft import PeftModel

#Constants
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLORES_CODES = {
    "asm_Beng": "as",
    "awa_Deva": "hi",
    "ben_Beng": "bn",
    "bho_Deva": "hi",
    "brx_Deva": "hi",
    "doi_Deva": "hi",
    "eng_Latn": "en",
    "gom_Deva": "kK",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "hne_Deva": "hi",
    "kan_Knda": "kn",
    "kas_Arab": "ur",
    "kas_Deva": "hi",
    "kha_Latn": "en",
    "lus_Latn": "en",
    "mag_Deva": "hi",
    "mai_Deva": "hi",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "bn",
    "mni_Mtei": "hi",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "hi",
    "sat_Olck": "or",
    "snd_Arab": "ur",
    "snd_Deva": "hi",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}
base_ckpt_dir = "ai4bharat/indictrans2-indic-en-dist-200M"
lora_ckpt_dir = "https://github.com/japjotsaggu-wai/NE_Indictrans2.git"

def split_sentences(input_text, lang):
    if lang == "eng_Latn":
        input_sentences = sent_tokenize(input_text)
    else:
        input_sentences = sentence_split(
            input_text, lang=FLORES_CODES[lang], delim_pat=DELIM_PAT_NO_DANDA
        )
    return [sent.replace("\xad", "") for sent in input_sentences]

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
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        inputs = tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        translations += tokenizer.batch_decode(
            generated_tokens.cpu().tolist(), skip_special_tokens=True
        )

        torch.cuda.empty_cache()

    return translations

def main():
    quantization = sys.argv[1] if len(sys.argv) > 3 else ""
    attn_implementation = sys.argv[2] if len(sys.argv) > 4 else "eager"

    ip = IndicProcessor(inference=True)

    tokenizer, lora_model = initialize_model_and_tokenizer(
        base_ckpt_dir, lora_ckpt_dir, quantization, attn_implementation
    )

    src_lang = "hin_Deva"  # Example source language
    tgt_lang = "eng_Latn"  # Target language

    input_text = "आपका स्वागत है।"
    input_sentences = split_sentences(input_text, src_lang)

    translations = batch_translate(input_sentences, src_lang, tgt_lang, lora_model, tokenizer, ip)
    print("Translations:", translations)

if __name__ == "__main__":
    main()
