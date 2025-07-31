from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from statistics import mean
import torch, time, json
import fire
from datasets import load_dataset
import os
from tqdm import tqdm
import random
import numpy as np


accelerator = Accelerator()

def apply_chat_template(model_name, prompts):
    
    '''load model & tokenizer'''
    if 'llama' in model_name.lower():
        # chat template: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/
        prompt_template = '''
                    <|begin_of_text|><|start_header_id|>system<|end_header_id|> {} <|eot_id|>
                    <|start_header_id|>user<|end_header_id|> {} \n## Instruction: {} <|eot_id|> 
                    <|start_header_id|>assistant<|end_header_id|>
                    '''
        
    elif 'mistral' in model_name.lower():
        # chat template: https://www.promptingguide.ai/models/mistral-7b
        # chat template: <s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
        prompt_template = '''
                    <s>[INST]system {} [/INST]
                    [INST]user {} \n## Instruction: {} [/INST]
                    [INST]assistant
                    '''    
    elif 'gemma' in model_name.lower():
        prompt_template = '''
            <bos><start_of_turn>system {} <end_of_turn>
            <bos><start_of_turn>user {}\n## Instruction: {} <end_of_turn>
            <start_of_turn>model
            '''    

    elif "phi" in model_name.lower():
        prompt_template = '''
             <|system|> {} <end>\n
            <|user|> {} \n## Instruction: {} <end>\n
            <|assistant|>
            '''    
    elif "qwen" in model_name.lower():
        # Reference: https://huggingface.co/Qwen/Qwen1.5-7B-Chat
        prompt_template = '''
            <|im_start|>system\n{}<|im_end|>\n
            <|im_start|>user\n{} \n## Instruction: {}<|im_end|>\n
            <|im_start|>assistant\n
            '''
    else:
        raise NotImplementedError
    
    system_prompt  = "You are a helpful assistant."
    user_prompt = "Please provide a response for the following instruction: "
    chat_prompts = [prompt_template.format(system_prompt, user_prompt, prompt) for prompt in prompts]
    
    return chat_prompts

# batch, left pad (for inference), and tokenize
def prepare_prompts(prompts, tokenizer, batch_size=16):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main(model_name_or_path = "/mnt/data1/jinlong/DPO-noisy-outputs/llama-3-1b-sft",
         eval_dataset_name = 'tatsu-lab/alpaca_eval',
         output_path = './model_outputs/',
         batch_size = 16,
    ):

    set_seed(seed=42)
    
    ## load dataset
    if eval_dataset_name == 'tatsu-lab/alpaca_eval':
        # eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")['eval'].select(list(range(20)))
        eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")['eval']
    else:
        raise NotImplementedError
    
    
    # load a base model and tokenizer
    model_tag = os.path.basename(model_name_or_path)   
     
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,   # 可以指定跳过量化的模块
        llm_int8_enable_fp32_cpu_offload=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,    
        device_map={"": accelerator.process_index},
        # torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)   
    tokenizer.pad_token = tokenizer.eos_token
    
    if  'qwen' in model_name_or_path.lower() and not tokenizer.bos_token:
        tokenizer.bos_token = "<|im_start|>"
        
    # print(tokenizer.chat_template)
    # import pdb;pdb.set_trace()
    if tokenizer.chat_template is not None:
        print("Add the chat template...")
        prompts_all= []
        for prompt in eval_set['instruction']:
            prompt_chat = [{"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": prompt},
                            ]
            
            prompt_chat = tokenizer.apply_chat_template(prompt_chat, tokenize=False, add_generation_prompt=True)   
            # prompt_chat = tokenizer.apply_chat_template(prompt_chat, tokenize=False)            
            prompts_all.append(prompt_chat)
        # import pdb;pdb.set_trace()
    else: ## add the chat template manually
        print("add the chat template manually...")
        prompts_all = apply_chat_template(model_name_or_path, eval_set['instruction'])

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    
    start=time.time()
    gpu_count = accelerator.state.num_processes
    progress_bar = tqdm(
        range(len(prompts_all) // (batch_size * gpu_count) + 1),
        disable=not accelerator.is_local_main_process,  #only the main process show progress
        desc="Generating AlpacaEval2's Responses")
    
    
    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(prompts_all) as prompts:
        results=dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=batch_size)

        for prompts_tokenized in prompt_batches:

            # outputs_tokenized = model.generate(**prompts_tokenized, do_sample=True, temperature=0.9, top_p=1.0, max_new_tokens=2048)
            outputs_tokenized = model.generate(
                **prompts_tokenized, 
                do_sample=True, 
                temperature=0.9, 
                top_p=1.0, 
                max_new_tokens=2048,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,       # 避免生成重复片段（可选）
                use_cache=True,
                )

            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

            # count and decode gen. tokens 
            num_tokens=sum([ len(t) for t in outputs_tokenized ])
            outputs=tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)

            # store in results{} to be gathered by accelerate
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens
            
            progress_bar.update(1)
            
        results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

    # collect results from all the GPUs
    results_gathered=gather_object(results)
    print("Finished response generation!")

    if accelerator.is_main_process:
        
        timediff=time.time()-start
        num_tokens=sum([r["num_tokens"] for r in results_gathered])
        
        generated_outputs = []
        for r in results_gathered:  ## merge each device's result
            generated_outputs.extend(r['outputs'])

        print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
        
        eval_set = eval_set.map(lambda x, idx: {
            'output': generated_outputs[idx],
            'generator': model_tag,
            }, 
            with_indices=True)
        

            
        model_outputs_file = os.path.join(output_path, model_tag, 'model_outputs_full.json')
        
        os.makedirs(os.path.dirname(model_outputs_file), exist_ok=True)

        # eval_set.to_json(model_outputs_file)
        
        ## save the eval set as a json object
        eval_set_dict = eval_set.to_dict()
        with open(model_outputs_file, 'w') as f:
            json.dump(eval_set_dict, f, indent=4)    
        
        print(f"Generations are stored in {model_outputs_file}! \n")
        


if __name__ == '__main__':
    fire.Fire(main)