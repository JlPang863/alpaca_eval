export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


eval_model="alpaca_eval_gpt4.1"

######################################################################################################
######################################################################################################


MODEL_LIST=(
# princeton-nlp/Llama-3-Instruct-8B-DPO
# princeton-nlp/Llama-3-Instruct-8B-SimPO
# princeton-nlp/Mistral-7B-Instruct-DPO
# princeton-nlp/Mistral-7B-Instruct-SimPO
    # llama-3-8b-instruct-mix-dpo
    # llama-3-8b-instruct-mix-dpo-loss
    # llama-3-8b-instruct-mix-dpo-rm
    # llama-3-8b-instruct-selective-dpo
    # mistral-7b-instruct-mix-dpo
    # mistral-7b-instruct-mix-dpo-loss
    # mistral-7b-instruct-mix-dpo-rm
    # mistral-7b-instruct-selective-dpo
    # mistral-7b-instruct-mix-dpo-lr1
    # mistral-7b-instruct-mix-dpo-lr2
    # llama-3-8b-base-mix-dpo-5-swap
    # llama-3-8b-base-cal-dpo
    # mistral-7b-base-cal-dpo
    # llama-3-8b-base-dpo-ches-0.5
    # llama-3-8b-base-dpo-ches-0.9
    # llama-3-8b-base-dpo-nll
    mistral-7b-base-dpo-nll
    # mistral-7b-base-dpo-ches-0.5
    # mistral-7b-base-dpo-ches-0.9
)

model_output_path="model_outputs_cl_instruct"

# response_file="model_outputs_20.json" ###test4
response_file="model_outputs_full.json" ###full


for MODEL in ${MODEL_LIST[@]}; do

echo "*** MODEL : $MODEL ***"

model_output_file="${model_output_path}/$(basename $MODEL)/${response_file}"

echo "*** current model output file: ${model_output_file} ***"

alpaca_eval --model_outputs $model_output_file \
--annotators_config $eval_model

done

# model_outputs_cl_instruct/Llama-3-Instruct-8B-DPO/model_outputs_full.json