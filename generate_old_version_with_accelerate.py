from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
import fire

class CustomDataset(Dataset):
    def __init__(self, dialogs):
        self.dialogs = dialogs

    def __getitem__(self, idx):
        dialog = self.dialogs[idx]
        return dialog, idx

    def __len__(self):
        return len(self.dialogs)


# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"  # Set your GPU IDs

def main(model_tag="pythia28_hh_selected_clean_subset",
         eval_dataset_name='tatsu-lab/alpaca_eval',
         data_path='./AlpacaEval2_responses/',
         batch_size=32):

    # Initialize Accelerator
    accelerator = Accelerator()

    # Load dataset
    if eval_dataset_name == 'tatsu-lab/alpaca_eval':
        eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")['eval'].select(list(range(10)))
    else:
        raise NotImplementedError

    # Load the model and tokenizer
    model_name_or_path = f"/mnt/data1/jinlong/compare_model_preferences/{model_tag}_model"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare instructions
    instructions = eval_set['instruction']

    # Create the custom dataset and dataloader
    dataset = CustomDataset(instructions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare model and dataloader with Accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    # List to store generated outputs
    generated_outputs = []

    # Loop through the dataloader and generate responses
    encoding_outputs = []
    indices = []

    model.eval()
    for batch in tqdm(dataloader, desc="Generating AlpacaEval2's Responses"):
        prompts, batch_indices = batch
        encodings = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(accelerator.device)
        # Generate responses without gradient calculation
        with torch.no_grad():
            outputs = model.generate(
                        **encodings, 
                        do_sample=True, 
                        temperature=0.9, 
                        top_p=1.0, 
                        max_new_tokens=2048)
            
        import pdb;pdb.set_trace()

        encoding_outputs.append(outputs)
        indices.append(batch_indices)
        
        # Decode the generated outputs
        # output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # generated_outputs.extend(output_texts)

    accelerator.wait_for_everyone()

    all_encoding_outputs= accelerator.gather(torch.cat(encoding_outputs, dim=0))
    all_indices = accelerator.gather(torch.cat(indices, dim=0))
    
    if accelerator.is_main_process:
        
        
        all_encoding_outputs = all_encoding_outputs.cpu().tolist()
        all_indices = all_indices.cpu().tolist()

        gathered_results = {}
        gathered_results = dict(zip(all_indices, all_encoding_outputs)) 
        
        sorted_results = sorted(gathered_results.items(), key=lambda x: x[0]) 
        
        sorted_encoding_outputs = [x[1] for x in sorted_results]
        import pdb;pdb.set_trace()
        generated_outputs = tokenizer.batch_decode(sorted_encoding_outputs, skip_special_tokens=True)

        
        # Map generated outputs to the eval_set dataset
        eval_set = eval_set.map(lambda x, idx: {
            'output': generated_outputs[idx][len(x['instruction']):],  # Only take the generated part
            'generator': model_tag,
            },
            with_indices=True)

        # Save the results
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        eval_set.to_json(os.path.join(data_path, f"{model_tag}.json"))
        print("Finished response generation!")
        print(f"Generations are stored in {os.path.join(data_path, f'{model_tag}.json')}")


if __name__ == '__main__':
    fire.Fire(main)
