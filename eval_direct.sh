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



################################################################################################
################################################################################################
# model_config="/home/jlpang/alpaca_eval/model_configs/configs/llama-3-1b-ours-merged.yaml"
# model_config="/home/jlpang/alpaca_eval/src/alpaca_eval/models_configs/Llama-3-Instruct-8B-SimPO/configs.yaml"
# model_config="/home/jlpang/alpaca_eval/model_configs/configs/llama-3-8b-ours4-6-sorted-score-diff-full-test.yaml"

# model_config="/home/jlpang/alpaca_eval/model_configs/configs/llama-3-8b-simpo-test.yaml"
model_config="/home/jlpang/alpaca_eval/model_configs/configs/llama-3-8b-dpo-test.yaml"

# # need a GPU for local models
alpaca_eval evaluate_from_model \
  --model_configs $model_config \
  --annotators_config $eval_model