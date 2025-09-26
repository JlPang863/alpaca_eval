# export CUDA_VISIBLE_DEVICES=7
# NUM_GPUS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

main_process_port=29517
root_path="/home/azureuser/cloudfiles/code/Users/jinlong.pang/QualityDPO/CL_DPO_outputs"

BATCH_SIZE=8
OUTPUT_PATH="./model_outputs_cl_instruct"

base_model=llama-3-8b
MODEL_LIST=(
    # mix-dpo-sorted-embedding-dist-1e06
    # mix-dpo-sorted-embedding-dist-3e07
    # mix-dpo-sorted-embedding-dist-5e07
    # mix-dpo-sorted-embedding-dist-8e07
    mix-dpo-sorted-llama-loss-1e06
    # mix-dpo-sorted-llama-loss-5e07
    # mix-dpo-sorted-llama-loss-8e07
    # mix-dpo-sorted-reward-1e06
    # mix-dpo-sorted-reward-5e07
    # mix-dpo-sorted-reward-8e07
)


for model_name_or_path in ${MODEL_LIST[@]}; do

        echo "*** model_name_or_path: ${base_model}-base-$model_name_or_path ***"
        accelerate launch \
            --num_processes $NUM_GPUS \
            --main_process_port $main_process_port \
            --mixed_precision bf16 \
            generate_response.py \
            --model_name_or_path $root_path/${base_model}-base-$model_name_or_path/checkpoint-336 \
            --batch_size $BATCH_SIZE \
            --output_path $OUTPUT_PATH \

done 

# huggingface-cli download princeton-nlp/Llama-3-Instruct-8B-DPO --local-dir /home/azureuser/cloudfiles/code/Users/jinlong.pang/QualityDPO/CL_DPO_outputs