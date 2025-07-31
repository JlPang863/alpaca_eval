export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 


eval_model="alpaca_eval_gpt4.1"

######################################################################################################
######################################################################################################

# base_model="llama-3-1b" 
# base_model="mistral-7b" 
BASE_MODELS=(
    llama-3-8b
    mistral-7b
)

# model_type='instruct'
model_type='base'
loss_type_list=(
    sft-chosen
    # org
    # dpo
    # simpo
    # selective-dpo
    # mix-dpo
    # dpo-identical-pairs-high-quality-2000
    # dpo-identical-pairs-low-quality-2000
    ) 

model_output_path="model_outputs_cl"

# response_file="model_outputs_20.json" ###test4
response_file="model_outputs_full.json" ###full


for base_model in ${BASE_MODELS[@]}; do

    for loss_type in ${loss_type_list[@]}; do

        # model_output_file="model_outputs/${base_model}-${loss_type}-merged/model_outputs_20.json"
        echo "*** loss type: $loss_type ***"
        echo "*** base model: $base_model ***"
        if [[ $loss_type == "org" ]]; then
            if [[ $base_model == "llama-3-8b" ]]; then
                model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
            elif [[ $base_model == "mistral-7b" ]]; then
                model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2"
            else
                echo "unknown base model"
            fi
            model_output_file="${model_output_path}/$(basename $model_name_or_path)/${response_file}"

        else

            model_output_file="${model_output_path}/${base_model}-${model_type}-${loss_type}/${response_file}"
        fi
        echo "*** current model output file: ${model_output_file} ***"

        alpaca_eval --model_outputs  $model_output_file \
            --annotators_config $eval_model

    done

done 
################################################################################################
################################################################################################
# model_config="/home/jlpang/alpaca_eval/model_configs/configs/llama-3-1b-ours-merged.yaml"
# model_config="/home/jlpang/alpaca_eval/src/alpaca_eval/models_configs/Llama-3-Instruct-8B-SimPO/configs.yaml"


# # need a GPU for local models
# alpaca_eval evaluate_from_model \
#   --model_configs $model_config \
#   --annotators_config $eval_model