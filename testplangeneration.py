# =============================================================================
# USE CASE: AI-Assisted Test Plan Generation
# =============================================================================
# LEARNING MECHANISM — OFFLINE FEEDBACK ONLY:
#   This model is trained exclusively through offline supervised fine-tuning
#   (SFT) on a curated, human-reviewed dataset. There is NO online reinforcement
#   learning, no real-time reward signals, and no live environment interaction.
#   Model weights are updated only when a new offline training run is explicitly
#   triggered by an authorised team member using reviewed training data.
#
# HUMAN REVIEW & APPROVAL REQUIREMENT:
#   All AI-generated test plans produced by this model MUST be reviewed and
#   approved by a qualified human engineer before being used in any testing
#   activity. Generated outputs must NOT be treated as final or authoritative
#   without explicit human sign-off. This applies to every inference call,
#   including automated pipeline executions.
# =============================================================================

import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import tqdm 
import torch
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import atexit, sys

print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))

# Handle proxies (same as your main script)
os.environ['HTTP_PROXY'] = "http://gateway.schneider.zscaler.net:80"
os.environ['HTTPS_PROXY'] = "http://gateway.schneider.zscaler.net:80"
os.environ['http_proxy'] = 'http://gateway.schneider.zscaler.net:80'
os.environ['https_proxy'] = 'http://gateway.schneider.zscaler.net:80'

# Disable SSL
os.environ['CURL_CA_BUNDLE'] = r"C:\Users\SESI019354\Downloads\ZscalerRootCA.pem"
os.environ['PYTHONHTTPSVERIFY'] = "0"
os.environ['REQUESTS_CA_BUNDLE'] = r"C:\Users\SESI019354\Downloads\ZscalerRootCA.pem"
os.environ['SSL_CERT_FILE'] = r"C:\Users\SESI019354\Downloads\ZscalerRootCA.pem"

#Loading model
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "./qa-assitant-lora"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

#Loading dataset
dataset = load_dataset("json", data_files=r"C:\Users\SESI019354\Desktop\Testplan generation\dataset.json")["train"]

def format_prompt(example):
    return f"""### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"""
# Disable multiprocessing in dataset.map to avoid ResourceTracker issues
dataset = dataset.map(lambda x:{"text": format_prompt(x)})

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation = True, max_length = 328, padding = "max_length")

dataset =dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

#Load Model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = bnb_config,
    device_map = "auto"
)

peft = LoraConfig(
    r=8,
    lora_alpha = 16,
    target_modules= ["q_proj", "k_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

#Training Arguments
training_args = TrainingArguments(
    output_dir = "./test_ai_model",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    num_train_epochs = 3,
    max_steps = 20,
    learning_rate = 2e-4,
    bf16 = True,
    logging_steps = 5,
    report_to = "none",
    save_strategy = "no",
    optim = "paged_adamw_8bit"
)

trainer = SFTTrainer(
    model = model,
    train_dataset=dataset,
    peft_config = peft,
    args = training_args
)

trainer.train()

trainer.save_model("./qa-assitant-lora")


atexit.register(lambda: sys.exit(0))

pipe = pipeline(model = "./qa-assitant-lora", tokenizer = tokenizer)

prompt = """### Instruction: \nGenerate a system test plan for Schneider Electric Ecostruxure Login page. \n\n### Response:\n"""
output = pipe(prompt, max_new_tokens=400)
print(output[0]["generated_text"])

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()