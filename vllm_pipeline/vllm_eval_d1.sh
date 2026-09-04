# greedy evaluation

# python greedy_eval_vllm.py --model_name llama3.1-8b-it --data_name triviaqa --format direct --device_map 0,1
# python greedy_eval_vllm.py --model_name llama3.1-8b-it --data_name triviaqa --format reflect2 --device_map 0,1
# python greedy_eval_vllm.py --model_name llama3.1-8b-it --data_name nq --format direct --device_map 0,1
# python greedy_eval_vllm.py --model_name llama3.1-8b-it --data_name nq --format reflect2 --device_map 0,1
# python greedy_eval_vllm.py --model_name llama3.1-8b-it --data_name webq --format direct --device_map 0,1
# python greedy_eval_vllm.py --model_name llama3.1-8b-it --data_name webq --format reflect2 --device_map 0,1

# python greedy_eval_vllm.py --model_name qwen2.5-7b-it --data_name triviaqa --format direct --device_map 0,1
# python greedy_eval_vllm.py --model_name qwen2.5-7b-it --data_name triviaqa --format reflect2 --device_map 0,1
# python greedy_eval_vllm.py --model_name qwen2.5-7b-it --data_name nq --format direct --device_map 0,1
# python greedy_eval_vllm.py --model_name qwen2.5-7b-it --data_name nq --format reflect2 --device_map 0,1
# python greedy_eval_vllm.py --model_name qwen2.5-7b-it --data_name webq --format direct --device_map 0,1
# python greedy_eval_vllm.py --model_name qwen2.5-7b-it --data_name webq --format reflect2 --device_map 0,1

# python greedy_eval_vllm.py --model_name mistral-8b-it --data_name nq --format direct --device_map 0,1
# python greedy_eval_vllm.py --model_name mistral-8b-it --data_name nq --format reflect2 --device_map 0,1
# python greedy_eval_vllm.py --model_name mistral-8b-it --data_name triviaqa --format direct --device_map 0,1
# python greedy_eval_vllm.py --model_name mistral-8b-it --data_name triviaqa --format reflect2 --device_map 0,1
# python greedy_eval_vllm.py --model_name mistral-8b-it --data_name webq --format direct --device_map 0,1
# python greedy_eval_vllm.py --model_name mistral-8b-it --data_name webq --format reflect2 --device_map 0,1


# sampling evaluation

# python sampling_eval_vllm.py --model_name llama3.1-8b-it --data_name triviaqa --format direct --device_map 1
# python sampling_eval_vllm.py --model_name llama3.1-8b-it --data_name triviaqa --format reflect2 --device_map 1
# python sampling_eval_vllm.py --model_name llama3.1-8b-it --data_name nq --format direct --device_map 1
# python sampling_eval_vllm.py --model_name llama3.1-8b-it --data_name nq --format reflect2 --device_map 1
# python sampling_eval_vllm.py --model_name llama3.1-8b-it --data_name webq --format direct --device_map 1
# python sampling_eval_vllm.py --model_name llama3.1-8b-it --data_name webq --format reflect2 --device_map 1

# python sampling_eval_vllm.py --model_name qwen2.5-7b-it --data_name triviaqa --format direct --device_map 1
# python sampling_eval_vllm.py --model_name qwen2.5-7b-it --data_name triviaqa --format reflect2 --device_map 1
# python sampling_eval_vllm.py --model_name qwen2.5-7b-it --data_name nq --format direct --device_map 1
# python sampling_eval_vllm.py --model_name qwen2.5-7b-it --data_name nq --format reflect2 --device_map 1
# python sampling_eval_vllm.py --model_name qwen2.5-7b-it --data_name webq --format direct --device_map 1
# python sampling_eval_vllm.py --model_name qwen2.5-7b-it --data_name webq --format reflect2 --device_map 1

# python sampling_eval_vllm.py --model_name mistral-8b-it --data_name nq --format direct --device_map 0,1
# python sampling_eval_vllm.py --model_name mistral-8b-it --data_name nq --format reflect2 --device_map 0,1
# python sampling_eval_vllm.py --model_name mistral-8b-it --data_name triviaqa --format direct --device_map 0,1
# python sampling_eval_vllm.py --model_name mistral-8b-it --data_name triviaqa --format reflect2 --device_map 0,1
# python sampling_eval_vllm.py --model_name mistral-8b-it --data_name webq --format direct --device_map 0,1
# python sampling_eval_vllm.py --model_name mistral-8b-it --data_name webq --format reflect2 --device_map 0,1


python greedy_eval_vllm.py --model_name olmo2-7b-it --data_name nq --format direct --device_map 1
# python greedy_eval_vllm.py --model_name olmo2-7b-it --data_name nq --format reflect2 --device_map 1
python greedy_eval_vllm.py --model_name olmo2-7b-it --data_name triviaqa --format direct --device_map 1
# python greedy_eval_vllm.py --model_name olmo2-7b-it --data_name triviaqa --format reflect2 --device_map 1
python greedy_eval_vllm.py --model_name olmo2-7b-it --data_name webq --format direct --device_map 1
# python greedy_eval_vllm.py --model_name olmo2-7b-it --data_name webq --format reflect2 --device_map 1

# python sampling_eval_vllm.py --model_name olmo2-7b-it --data_name nq --format direct --device_map 1
# python sampling_eval_vllm.py --model_name olmo2-7b-it --data_name nq --format reflect2 --device_map 1
# python sampling_eval_vllm.py --model_name olmo2-7b-it --data_name triviaqa --format direct --device_map 1
# python sampling_eval_vllm.py --model_name olmo2-7b-it --data_name triviaqa --format reflect2 --device_map 1
# python sampling_eval_vllm.py --model_name olmo2-7b-it --data_name webq --format direct --device_map 1
# python sampling_eval_vllm.py --model_name olmo2-7b-it --data_name webq --format reflect2 --device_map 1