import torch
import torch.optim as optim
from transformers import BertTokenizer
import os
import random
import tkinter as tk
from mixture_of_experts import MoE
import wandb
import random
import math
import gzip
import torch.optim as lr_scheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.profiler import profile, ProfilerActivity
import time
import random
import torch
from rotary_embedding_torch import RotaryEmbedding


torch.set_default_device('cuda:0')


torch.set_float32_matmul_precision('high')

global d_model
d_model = 768
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# repeat(batch_size,1,1)
class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
        # self.tokenizer.add_special_tokens({"additional_special_tokens": ["[MASK]"]})
        # print(self.tokenizer.vocab_size)
    def tokenize_string(self, string):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(string))
    def decode(self,string):
        return self.tokenizer.decode(string)
    
TOKEN_BYTES = 4

class Preprocessing:
    def __init__(self):
        
        self.tok = Tokenizer()

    def preprocessing(self):
  
        

        a = open('some_code.txt','w')
        z = 0
        with gzip.open('file-000000000015.json.gz', 'rb') as f:
            file_content = f.read()
            
            
            for line in file_content:
                a.write(str(line))
            
            
            a.close()
    def data_preperation(self):
    
        
        tokens = 0
        lines = 0
        
        with open('reddit_binary.bin', mode='wb') as file:

            with open('complete_comments3.txt', "r", encoding="utf-8") as input_file:
                bytes_read = 0
                for line in input_file:
                    # print(line)

                    # print(line)
                    bytes_read += len(line)
                    # if line != '\n':
                    tokenized_line = self.tok.tokenize_string(line)
                    for i in tokenized_line:
                        file.write(i.to_bytes(TOKEN_BYTES, 'big'))
                        
                        tokens += 1
                    lines += 1
                    
    

    def fetch_training_batch(self, B, S):
        file_name = "tokenized_giant.bin"
        f = open(file_name,'rb')
        total_bytes = os.path.getsize(file_name)
      
        total_tokens = total_bytes // 4
        input_batch = []
        target_batch = []
        for i in range(B):
            start_token = random.randint(0, total_tokens-S)
            
            start_byte = start_token * TOKEN_BYTES
            f.seek(start_byte)
            tokens_target = []
            tokens_input = []
            data = f.read(S*TOKEN_BYTES)
            arr = []
          
            
            for z in range(S):
                if z == 0:
                    tokens_input.append(50256)
                    tokens_target.append(int.from_bytes(data[z*TOKEN_BYTES:(z+1)*TOKEN_BYTES],"big"))
                else:
                    list1 = int.from_bytes(data[(z-1)*TOKEN_BYTES:(z)*TOKEN_BYTES],"big")  
                    tokens_input.append(list1)
                    tokens_target.append(int.from_bytes(data[z*TOKEN_BYTES:(z+1)*TOKEN_BYTES],"big"))
            
            target_batch.append(tokens_target)
            input_batch.append(tokens_input)
            

        target_batch = torch.tensor(target_batch)
        input_batch = torch.tensor(input_batch)

        return input_batch,target_batch


def b16c(param):
    return param.to(torch.bfloat16).cuda()
    

def Positional_Encoding(tensor_BSE):
    n = 10000
    B,S,E = tensor_BSE.shape
    tensor_BSE = tensor_BSE.to(device)
    e = E
    z = torch.arange(-0.5,E//2-0.5,0.5).to(device)
    # print(z)
    range_e = torch.ceil(z)
    # print(range_e)
    range_e = range_e.repeat(S,1)
    range_e.to(device)
    range_E = torch.arange(1,S+1).to(device)
    i_e = range_e/E
    exponent_e = 2*i_e
    denom_e = torch.pow(n,exponent_e).to(device)
    a = range_E.expand(d_model,S).T.to(device)
    result = a/denom_e
    result[:, 1::2] = torch.cos(result[:, 1::2])
    result[:,::2] = torch.sin(result[:,::2])
    return result.repeat(B,1,1) + tensor_BSE

def norm(x_BSE):
    # print(x_BSE)
    square = torch.square(x_BSE)
    sum_BS1 = torch.mean(square, 2, keepdim=True)
    square_root_BS1 = torch.sqrt(sum_BS1)
    return x_BSE/square_root_BS1

class MMHA(torch.nn.Module):
    def __init__(self, Heads):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Kquerys = d_model//Heads
   
        self.w_EHK_k = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_v = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_q = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_o = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.rotary_emb = RotaryEmbedding(dim = self.Kquerys)
        
    def attention(self,x_BSE):
        x_BSE = x_BSE.to(self.device)
        # self.w_EHK_q = torch.nn.Parameter(self.rotary_emb.rotate_queries_or_keys(self.w_EHK_q))
        # self.w_EHK_k = torch.nn.Parameter(self.rotary_emb.rotate_queries_or_keys(self.w_EHK_k))
        """
        B: Batch Size
        H: # of Heads
        S: Sequence length
        K: # of querys
        E: Embedding dimention
        """
        x_BSE = b16c(x_BSE)
        query_BHSK = torch.einsum('BSE,EHK->BHSK',b16c(x_BSE),b16c(self.w_EHK_q))
        key_BHMK = torch.einsum('BSE,EHK->BHSK',b16c(x_BSE),b16c(self.w_EHK_k))
        value_BHMK = torch.einsum('BSE,EHK->BHSK',b16c(x_BSE),b16c(self.w_EHK_v))
        # print(query_BHSK.shape, key_BHMK.shape, value_BHMK.shape)

        query_BHSK = self.rotary_emb.rotate_queries_or_keys(query_BHSK)
        key_BHMK = self.rotary_emb.rotate_queries_or_keys(key_BHMK)
        
        logits_BSHM = torch.einsum('BHSK,BHMK->BSHM',b16c(query_BHSK), b16c(key_BHMK)) / math.sqrt(self.Kquerys)
        
        B, S, H, M = logits_BSHM.shape
        query_pos_1S11 = torch.arange(S, device=logits_BSHM.device).view(1, S, 1, 1)
        memory_pos_111M = torch.arange(M, device=logits_BSHM.device).view(1, 1, 1, M)
        visiBSE_1S1M = query_pos_1S11 >= memory_pos_111M
        mask_1S1M = torch.where(visiBSE_1S1M, 0, -torch.inf)
        logits_BSHM = logits_BSHM + mask_1S1M
        softmax_BSHM = torch.softmax(logits_BSHM, dim=3)
 
        output_BSHK = torch.einsum('BSHM,BHMK->BSHK',b16c(softmax_BSHM), b16c(value_BHMK))

        a = torch.einsum('BSHK,EHK->BSE',b16c(output_BSHK.to(self.device)),b16c(self.w_EHK_o))

        return a.to(self.device)

class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ffn_size = (d_model*8)//3
        # self.device = torch.devicecpu")
        self.ffn = ffn_size
        self.w_1_EF = torch.nn.Parameter(torch.randn(d_model,self.ffn).to(device) * (d_model ** -0.5))
        self.b_1_1F = torch.nn.Parameter(torch.randn(1,self.ffn).to(device) * (d_model ** -0.5))
        self.w_2_FE = torch.nn.Parameter(torch.randn(d_model,self.ffn).to(device) * (d_model ** -0.5))
        self.b_2_1E = torch.nn.Parameter(torch.randn(1,self.ffn).to(device) * (d_model ** -0.5))
        self.w_3_EF = torch.nn.Parameter(torch.randn(self.ffn,d_model).to(device) * (d_model ** -0.5))
        self.b_3_1E = torch.nn.Parameter(torch.randn(1,d_model).to(device) * (d_model ** -0.5))
        self.a = torch.nn.SiLU()
    def feed_forward(self,tensor_BSE):
        # tensor_BSE = tensor_BSE.to(self.device)
        out1 = self.a(b16c(tensor_BSE).to(device) @ b16c(self.w_1_EF).to(device) + b16c(self.b_1_1F).to(device))
        out2 = b16c(tensor_BSE) @ b16c(self.w_2_FE).to(device) + b16c(self.b_2_1E).to(device)
        z = torch.mul(out1,out2)
        
        return b16c(z) @ b16c(self.w_3_EF) + self.b_3_1E
    

class Decoder(torch.nn.Module):
    def __init__(self, Heads, ffn1):
        super().__init__()
      
        
        self.mha = MMHA(Heads)
        self.ffn = FeedForward()
    def forward(self, x_BSE):
        
        x_BSE = x_BSE + self.mha.attention(norm(b16c(x_BSE)))
        output = self.ffn.feed_forward(norm(x_BSE))
        x_BSE = x_BSE + output
        return x_BSE
    
class Transformer(torch.nn.Module):

    def __init__(self, writing, sampling):
        
        super().__init__()
        self.length_of_sentence = 256
        self.decoder_layers_num = 8
        self.Batch_size = 58
        self.Heads = 12

        self.lr = 0.005
        self.writing = writing
        self.step_num = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_iterations = 100000000000
        self.writing = writing
        self.Vocab_size = 50258

        self.embedding_matrix_VE = torch.nn.Parameter(torch.randn((self.Vocab_size, d_model)).to(self.device))
        self.ffn_size = d_model
        self.decoder_layers = torch.nn.ModuleList([Decoder(self.Heads, self.ffn_size) for i in range(self.decoder_layers_num)])
        self.final_linear = torch.nn.Linear(d_model, self.Vocab_size, device=self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
  
        self.preprocessing = Preprocessing()
        self.tokenizer = Tokenizer()
        num_parameters = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {num_parameters}")
        if sampling:
            saved_model_path = "Models/model.more_training_24layer_1540000"
            if os.path.exists(saved_model_path):
                print(saved_model_path)
                self.load_state_dict(torch.load(saved_model_path))
                torch.load(saved_model_path)

    def make_batch(self):
        return self.preprocessing.fetch_training_batch(self.Batch_size, self.length_of_sentence)
        
    def forward(self, decoder_inputs_BS):
        # take the un-embedded decoder input and put it to the device
        decoder_inputs_BS = decoder_inputs_BS.to(self.device
       
        embedded_inputs_BSE = torch.nn.functional.embedding(decoder_inputs_BS, self.embedding_matrix_VE)
        #forward through the decoder layers
        positionally_encoded_inputs_BSE = embedded_inputs_BSE
        for i in self.decoder_layers:
            positionally_encoded_inputs_BSE = i.forward(positionally_encoded_inputs_BSE)
        # return the final output of the decoder layers
        return norm(positionally_encoded_inputs_BSE)

    def training_step(self,i,training,input_sentence, lr):
        
            
        # make the batch of inputs for the model
        if training:
            inputs_BS, targets_BS = self.make_batch()
            
            inputs_BS = inputs_BS.to(self.device)
            targets_BS = targets_BS.to(self.device).float()
        
        else:
            inputs_BS = input_sentence.to(self.device)

        
        # put the batch through the forward to produce the outputs of the model
 
        logits_BSE = self.forward(inputs_BS).to(self.device)
        logits_BSV = self.final_linear(logits_BSE).to(self.device)
        logits_flat = logits_BSV.reshape(-1, logits_BSV.size(-1)).to(self.device)  # Reshape to (Batch size * Sequence Length, Vocab size)\
        if training:
            target_flat = targets_BS.to(torch.int64).reshape(-1).to(self.device)
        

        if training == False:
            # temperature = (int(global_slider_value)*1.5)/100 if int(global_slider_value) > 0 else 0.01
            temperature = 0.9
        
            probabilities = torch.nn.functional.softmax(logits_flat[-1] / temperature, dim=0)
            predicted = torch.multinomial(probabilities, 1).item()
         

            
            return predicted

        loss_B = torch.nn.functional.cross_entropy(logits_flat, target_flat, reduction='none')

        loss_B = torch.mean(loss_B)
        self.optimizer.zero_grad() 
        loss_B.backward()# Zero gradients
          # Compute gradients
        self.optimizer.step()
        return loss_B.item()

    def train_loop(self, writing):
            
        i = self.step_num
        losses = []
    
        update_interval = 1000
        save_interval = 2000
        
        saved_model_path = "none"
        print(saved_model_path)
        model_name = "S_Models/model.small_pretrained_model_e_"
        if os.path.exists(saved_model_path):
            print("EXISTS")
            self.load_state_dict(torch.load(saved_model_path))
            state_dict = torch.load(saved_model_path)
            print(f"{state_dict.keys()=}")

        if writing:
            wandb.init(
            
                project="Small_Pretrained_Models",
         
                config={
                
                "learning_rate": self.lr,
                "batch_size" : self.Batch_size,
                "decoder_layers": self.decoder_layers_num,
                "sentence_length": self.length_of_sentence,
                "d_model": d_model,
                "new_thing":"nothing",
                "saved_model_path":saved_model_path,
                "model_name":model_name,

                }
            )
        import time
        z = self.step_num

            
        start = time.time()
        while(self.step_num < self.training_iterations):

            average = 500
         
            i = self.step_num
 
            lr = self.lr / max(i, 100) ** 0.5 * min(1.0,i/100)
            for g in self.optimizer.param_groups:
                g['lr'] = lr 
            
            if i % save_interval == 0 and i > z+2:
                print("saving...")
                write_model_path = model_name+str(i)
                torch.save(self.state_dict(), write_model_path)

            

            # prof.step()
            start = time.time()
            a = self.training_step(i,5,0,lr=self.lr)
            torch.cuda.synchronize()
            end = time.time()

            if i % average == 0 and i >= 2:
             
                print(f"step {i}",f"loss {a}",f"lr {self.optimizer.param_groups[0]['lr']}", f"avg_loss={sum(losses)/average}", f"time={end-start}")
            
                losses = []
        
            if writing:
                wandb.log({"loss": a})

            # print("time=",(end-start))
            losses.append(a)
    
            self.step_num += 1
        end = time.time()
        print("average_time:",(end-start)/self.training_iterations)
        

    def sample(self,a):
        
        
        sentence_length = 100
        tokens1 = self.tokenizer.tokenize_string("<|endoftext|>") + self.tokenizer.tokenize_string(a)
        pure_generation = []
        for i in range(sentence_length):     
            tokens2 = torch.tensor(tokens1).unsqueeze(0)

            b = self.training_step(0,False,tokens2,0)
            tokens1.append(b)
            pure_generation.append(b)
        a = ''.join(self.tokenizer.decode(tokens1))
        return a

    def GUI_sample(self):
        root = tk.Tk()
        root.title("Input Box Example")
        root.geometry("400x300")
        text_widget = tk.Text(root, width=100, height=10)
        text_widget.pack()
        

        def on_button_click():
            text_widget.delete("1.0", tk.END)
            entered_text = entry.get() 
            print(entered_text)
            #do stuff here
            text_widget.insert(tk.END, self.sample(entered_text))



        entry = tk.Entry(root, width=125)
        entry.pack()


        button = tk.Button(root, text="Submit", command=on_button_click)
        button.pack()


        root.mainloop()
    

a = Transformer(False, False)
a.train_loop(True)
