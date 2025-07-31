from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import torch
import fire
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, dialogs):
        self.dialogs = dialogs

    def __getitem__(self, idx):
        dialog = self.dialogs[idx]
        return dialog, idx

    def __len__(self):
        return len(self.dialogs)
    
    
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

def main(model_tag = "pythia28_hh_selected_clean_subset",
         eval_dataset_name = 'tatsu-lab/alpaca_eval',
         data_path = './AlpacaEval2_responses/',
         batch_size = 32,
    ):

    if eval_dataset_name == 'tatsu-lab/alpaca_eval':
        eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")['eval']
    else:
        raise NotImplementedError

    # model_tag = "pythia28_hh_selected_clean_subset"
    model_name_or_path = f"/mnt/data1/jinlong/compare_model_preferences/{model_tag}_model"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token


    ### pythia model does not need additional prompts
    # instructions = []
    # for example in eval_set['instruction']:
    #     temp = "Q: " +  example + "\nA: "
    #     instructions.append(temp)
        
    instructions = eval_set['instruction']

    dataset = CustomDataset(instructions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    generated_outputs = []

    for batch in tqdm(dataloader, desc="Generating AlpacaEval2's Responses"):
        prompts, indices = batch
        encodings = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**encodings, do_sample=True, temperature=0.9, top_p=1.0, max_new_tokens=2048)

        output_texts =  tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_outputs.extend(output_texts)

    print("Finished response generation!")

    eval_set = eval_set.map(lambda x, idx: {
        'output': generated_outputs[idx][len(x['instruction']):],
        'generator': model_tag,
        }, 
        with_indices=True)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    eval_set.to_json(data_path + f"{model_tag}.json")
    
    print(f"Generations are stored in {os.path.join(data_path, f'{model_tag}.json')}")


if __name__ == '__main__':
    fire.Fire(main)