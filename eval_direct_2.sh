export CUDA_VISIBLE_DEVICES=2


eval_model="alpaca_eval_gpt4.1"
# model_config="/home/jlpang/alpaca_eval/model_configs/configs/llama-3-1b-ours-merged.yaml"
# model_config="/home/jlpang/alpaca_eval/src/alpaca_eval/models_configs/Llama-3-Instruct-8B-SimPO/configs.yaml"
# model_config="/home/jlpang/alpaca_eval/model_configs/configs/llama-3-8b-ours4-6-sorted-score-diff-full-test.yaml"

# model_config="/home/jlpang/alpaca_eval/model_configs/configs/llama-3-8b-simpo-test.yaml"
model_config="/home/jlpang/alpaca_eval/model_configs/configs/llama-3-8b-dpo-sorted-llama-full-ckpt-191.yaml"

# # need a GPU for local models
alpaca_eval evaluate_from_model \
  --model_configs $model_config \
  --annotators_config $eval_model