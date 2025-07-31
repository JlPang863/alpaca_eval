export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

# eval_model="alpaca_eval_gpt4_turbo_fn"
# eval_model="alpaca_eval_gpt4o_mini"
# eval_model="alpaca_eval_gpt4"
# eval_model="alpaca_eval_gpt4_turbo_fn_new"
# eval_model="oasst_pythia_12b"
# eval_model="alpaca_eval_gpt4_fn"

# eval_model="weighted_alpaca_eval_gpt-4o-mini-2024-07-18"
# eval_model="Self-taught-llama3.1-70B-dpo"
# eval_model="alpaca_eval_cot_gpt4_turbo_fn"
# eval_model="alpaca_eval_clf_gpt4_turbo"
# eval_model="alpaca_eval_gpt_o1"

# eval_model="gpt35_turbo_instruct" ## default use now
eval_model="alpaca_eval_gpt4.1"

######################################################################################################
######################################################################################################

base_model="llama-3-8b"

loss_type_list=(
    # dpo-full-eval
    ours4-6-sorted-score-diff-full-eval
    ) 
# ckpt_list=(38 76 114 152 190 228 266 304 342 382)
# ckpt_list=(76 152 228 304 382)
# ckpt_list=(38 114 190 266 342)

ckpt_list=(76)

model_output_path="model_outputs_cl_checkpoint_final"

# response_file="model_outputs_20.json" ###test4
response_file="model_outputs_full.json" ###full


for loss_type in ${loss_type_list[@]}; do


    for ckpt in ${ckpt_list[@]}; do
    echo "*** base model: $base_model ***"
    echo "*** loss type: $loss_type ***"
    echo "*** ckpt: $ckpt ***"

    model_output_file="${model_output_path}/${base_model}-${loss_type}/checkpoint-${ckpt}/${response_file}"

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