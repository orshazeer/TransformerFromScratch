from peft import LoraConfig
from transformers import AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset
import transformers
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
from datasets import load_from_disk
import torch
import json


class SFTModel():
    def __init__(self):
        self.model_name = "meta-llama/Meta-Llama-3.1-8B"
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        self.dataset = load_dataset("orshazeer/accepted-rejected-2")

    def quantize_with_adapters(self):
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model_4bit = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            quantization_config=quantization_config, 
            torch_dtype=torch.float16
            
        )   
      
        lora_config = LoraConfig(
            r=16,            
            lora_alpha=64, 
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.1,  
            bias="none"      
        )
       
    
        self.model_with_adapters = get_peft_model(model_4bit, lora_config)

    def load_quantized_model(self):
   
        self.model = AutoModelForCausalLM.from_pretrained(self.save_directory, device_map = "cpu")
      
        # for name, param in self.model.named_parameters():
        #     print(f"{name}: {param.dtype}")
        #     print(f"{param.requires_grad}")
        # print("Is the model quantized:", hasattr(self.model, 'quantize'))


    def tokenize_function(self, examples):
        new_examples = {
            "input_ids": [],
            "attention_mask": [], 

        }
        for chosen, rejected in zip(examples["accepted"], examples["rejected"]):

            tokenized_chosen = self.tokenizer(
            chosen,
            padding="max_length",
            max_length=2048,
            truncation=True,
            )


            new_examples["input_ids"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask"].append(tokenized_chosen["attention_mask"])


        return new_examples
    
    def save_tokenized_data(self,tokenized_data, file_path):
 
        tokenized_list = []
        for example in tokenized_data:
            tokenized_list.append({
                "input_ids": example["input_ids"],  
                "attention_mask": example["attention_mask"]  
            })
                

       
        with open(file_path, 'w') as f:
            json.dump(tokenized_list, f)
    
    def load_tokenized_data(self, file_path):


        with open(file_path, 'r') as f:
            tokenized_list = json.load(f)
       
            input_ids = torch.tensor([example["input_ids"] for example in tokenized_list])
            attention_masks = torch.tensor([example["attention_mask"] for example in tokenized_list])

            tokenized_data = {
            "input_ids": input_ids,
            "attention_mask": attention_masks
            }

            return tokenized_data

    def tokenized_data_mapping(self):

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenized_data = self.dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=4,
            
        )
  
        self.tokenized_data = self.tokenized_data.remove_columns(["accepted", "rejected"])
     
        max_len = 2048
        self.tokenized_data = self.tokenized_data.filter(
            lambda x: len(x["input_ids"]) <= max_len
            and len(x["attention_mask"]) <= max_len
        )

        dataset = {'input_ids' : self.tokenized_data['train']['input_ids'],
                   'attention_mask' : self.tokenized_data['train']['attention_mask']
                   }
    
        self.save_tokenized_data(self.tokenized_data['train'], 'tokenized_sft_train.json')
        self.dataset.save_to_disk("/home/or/Desktop/mapped_SFT_dataset")

        
  
    def training(self):
        seq_len = 768

        self.dataset = self.load_tokenized_data('tokenized_sft_train.json')

       
        self.data_collator = transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False, pad_to_multiple_of=seq_len)

        self.tokenizer.pad_token = self.tokenizer.eos_token
   
        self.model = self.model_with_adapters

        self.model.train()


        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
    
        lr = 0.0001
        batch_size = 1
        num_epochs = 10000000
        training_args = transformers.TrainingArguments(
            output_dir= "llama-8b-SFT-1",
            learning_rate=lr, 
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0,
            logging_strategy="no",
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=100,
            logging_steps=1,
            load_best_model_at_end=False,
            gradient_accumulation_steps=4,
            # gradient_checkpointing=True,
            
            warmup_steps=2,
            fp16=True,
            optim="paged_adamw_8bit",
            
            
        )

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.dataset['input_ids'],
            args=training_args,
            data_collator=self.data_collator
        )

        self.model.config.use_cache = False  
        import os
        os.environ["WANDB_DISABLED"] = "true"

        torch.cuda.empty_cache()
        trainer.train()
   

        self.model.config.use_cache = True
    
    def infrence(self, convo):

        inputs = self.tokenizer(convo, return_tensors="pt")

        outputs = self.model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=768, temperature=0.9)
        
        return self.tokenizer.batch_decode(outputs)[0]

    def infrence_loop(self):
        self.model_name1 = "llama-8b-SFT-1/checkpoint-27100"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )

        self.model_five = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="cuda",trust_remote_code=False, revision="main", quantization_config=quantization_config)

        self.model = PeftModel.from_pretrained(self.model_five, self.model_name1, device_map="cuda")
        self.model = self.model.to('cuda')
        self.model.eval()
        while True:
            
            a = input("Enter your system:\n\n")
            b = input("\n\nEnter you prompt:\n\n")
            prompt = "[SYST]"+a+"\n[QUEST]"+b+"\n[ANS]"
            output = self.infrence(prompt)
            print(output)
