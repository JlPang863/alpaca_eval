from accelerate import Accelerator
from accelerate.utils import gather_object
from statistics import mean
import torch, time, json
import fire
from datasets import load_dataset
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams 
from transformers import AutoModelForCausalLM, AutoTokenizer

accelerator = Accelerator()

def apply_chat_template(model_name, prompts):
    '''load model & tokenizer'''
    if 'llama' in model_name.lower():
        prompt_template = '''
                    <|begin_of_text|><|start_header_id|>system<|end_header_id|> {} <|eot_id|>
                    <|start_header_id|>user<|end_header_id|> {} \n## Instruction: {} <|eot_id|> 
                    <|start_header_id|>assistant<|end_header_id|>
                    '''
        
    elif 'mistral' in model_name.lower():
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
    else:
        raise NotImplementedError
    
    system_prompt  = "You are a helpful assistant."
    user_prompt = "Please provide a response for the following instruction: "
    chat_prompts = [prompt_template.format(system_prompt, user_prompt, prompt) for prompt in prompts]
    
    return chat_prompts

def main(model_name_or_path = "/mnt/data1/jinlong/DPO-noisy-outputs/llama-3-1b-sft",
         eval_dataset_name = 'tatsu-lab/alpaca_eval',
         output_path = './model_outputs/',
         batch_size = 16,
         use_vllm = True,  # ✅ 新增参数，控制是否使用 vLLM
         max_new_tokens = 2048
    ):

    ## load dataset
    if eval_dataset_name == 'tatsu-lab/alpaca_eval':
        eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")['eval']
    else:
        raise NotImplementedError
    
    # load tokenizer
    model_tag = os.path.basename(model_name_or_path)    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)   
    tokenizer.pad_token = tokenizer.eos_token
    
    # 处理 Prompt 格式
    if tokenizer.chat_template is not None:
        print("Add the chat template...")
        prompts_all= []
        for prompt in eval_set['instruction']:
            prompt_chat = [{"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": prompt},
                            ]
            
            prompt_chat = tokenizer.apply_chat_template(prompt_chat, tokenize=False)            
            prompts_all.append(prompt_chat)
    else: ## add the chat template manually
        print("add the chat template manually...")
        prompts_all = apply_chat_template(model_name_or_path, eval_set['instruction'])

    # **使用 vLLM 进行推理**
    if use_vllm:
        print("🚀 Using vLLM for fast inference...")
        llm = LLM(model=model_name_or_path, dtype="bfloat16", tensor_parallel_size=torch.cuda.device_count())
        sampling_params = SamplingParams(temperature=0.9, top_p=1.0, max_tokens=max_new_tokens, stop=["\n"])
        
        # vLLM 生成推理结果
        outputs = llm.generate(prompts_all, sampling_params)
        
        # 解析结果
        generated_outputs = [output.outputs[0].text.strip() for output in outputs]

    else:
        # default
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,    
            device_map={"": accelerator.process_index},
            torch_dtype=torch.bfloat16)
        
        accelerator.wait_for_everyone()    
        start=time.time()
        gpu_count = accelerator.state.num_processes
        progress_bar = tqdm(
            range(len(prompts_all) // (batch_size * gpu_count) + 1),
            disable=not accelerator.is_local_main_process,  
            desc="Generating Responses")
        
        generated_outputs = []
        with accelerator.split_between_processes(prompts_all) as prompts:
            prompt_batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
            for prompt_batch in prompt_batches:
                tokenized_prompts = tokenizer(prompt_batch, return_tensors="pt", padding='longest', truncation=False, pad_to_multiple_of=8).to("cuda") 
                
                outputs_tokenized = model.generate(**tokenized_prompts, do_sample=True, temperature=0.9, top_p=1.0, max_new_tokens=max_new_tokens)
                
                outputs_decoded = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
                generated_outputs.extend(outputs_decoded)
                
                progress_bar.update(1)
        
        print("Finished response generation!")

    if accelerator.is_main_process:
        timediff=time.time()-start if not use_vllm else 0
        print(f"Time elapsed: {timediff}, Generated samples: {len(generated_outputs)}")

        eval_set = eval_set.map(lambda x, idx: {
            'output': generated_outputs[idx],
            'generator': model_tag,
            }, with_indices=True)

        model_outputs_file = os.path.join(output_path, model_tag, 'model_outputs_full.json')
        os.makedirs(os.path.dirname(model_outputs_file), exist_ok=True)

        # Save the JSON output
        with open(model_outputs_file, 'w') as f:
            json.dump(eval_set.to_dict(), f, indent=4)    
        
        print(f"Generations are stored in {model_outputs_file}! \n")

if __name__ == '__main__':
    fire.Fire(main)
