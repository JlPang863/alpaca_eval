export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


eval_model="alpaca_eval_gpt4.1"

######################################################################################################
######################################################################################################


base_model=llama-3-8b

MODEL_LIST=(
    # mix-dpo-sorted-embedding-dist-1e06
    # mix-dpo-sorted-embedding-dist-3e07
    # mix-dpo-sorted-embedding-dist-5e07
    # mix-dpo-sorted-embedding-dist-8e07
    # mix-dpo-sorted-llama-loss-1e06
    # mix-dpo-sorted-llama-loss-5e07
    # mix-dpo-sorted-llama-loss-8e07
    # mix-dpo-sorted-reward-1e06
    # mix-dpo-sorted-reward-5e07
    # mix-dpo-sorted-reward-8e07
    # checkpoint-336
    # mix-dpo-sorted-llama-loss-1e06-ckpt-336
    # mix-dpo-sorted-reward-3e07
    # mix-dpo-sorted-llama-loss-1e06
    # mix-dpo-sorted-reward-6e07
    # mix-dpo-sorted-reward-4e07
    # mix-dpo-sorted-reward-7e07
    # mix-dpo-sorted-llama-loss-1e06-half
    # dpo-sorted-llama-loss-1e06
    # dpo-sorted-llama-loss-1e06-new
    mix-dpo-sorted-score-1e06
    mix-dpo-sorted-score-5e07
)


model_output_path="model_outputs_cl_instruct"

# response_file="model_outputs_20.json" ###test4
response_file="model_outputs_full.json" ###full


for MODEL in ${MODEL_LIST[@]}; do

echo "*** MODEL: $MODEL ***"

model_output_file="${model_output_path}/${base_model}-base-$(basename $MODEL)/${response_file}"

echo "*** current model output file: ${model_output_file} ***"

alpaca_eval --model_outputs $model_output_file \
--annotators_config $eval_model

done

# model_outputs_cl_instruct/Llama-3-Instruct-8B-DPO/model_outputs_full.json