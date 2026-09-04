# 모든 경우에 해당하는 vllm 생성


# 1. llama
# python greedy_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name triviaqa --format direct --device_map 2,3
# python greedy_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format direct --device_map 2,3
# python greedy_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name webq --format direct --device_map 2,3
# python sampling_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name triviaqa --format direct --device_map 2,3
# python sampling_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format direct --device_map 2,3
# python sampling_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name webq --format direct --device_map 2,3

# 2. qwen
# python greedy_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name triviaqa --format direct --device_map 2,3
# python greedy_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format direct --device_map 2,3
# python greedy_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name webq --format direct --device_map 2,3
# python sampling_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name triviaqa --format direct --device_map 2,3
# python sampling_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format direct --device_map 2,3
# python sampling_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name webq --format direct --device_map 2,3

# 3. mistral(정식명칭은 ministral)
# python greedy_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name triviaqa --format direct --device_map 2,3
# python greedy_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format direct --device_map 2,3
# python greedy_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name webq --format direct --device_map 2,3
# python sampling_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name triviaqa --format direct --device_map 2,3
# python sampling_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format direct --device_map 2,3
# python sampling_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name webq --format direct --device_map 2,3


# 1. llama
# python greedy_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 2,3
# python greedy_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 2,3
# python greedy_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name webq --format reflect2 --device_map 2,3
# python sampling_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 2,3
# python sampling_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 2,3
# python sampling_vllm_dev.py --model_name llama3.1-8b-it --num_gen 5 --data_name webq --format reflect2 --device_map 2,3

# 2. qwen
# python greedy_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 2,3
# python greedy_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format reflect2 --device_map 2,3
# python greedy_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name webq --format reflect2 --device_map 2,3
# python sampling_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 2,3
# python sampling_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format reflect2 --device_map 2,3
# python sampling_vllm_dev.py --model_name qwen2.5-7b-it --num_gen 5 --data_name webq --format reflect2 --device_map 2,3

# 3. mistral(정식명칭은 ministral)
# python greedy_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 2,3
# python greedy_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 2,3
# python greedy_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name webq --format reflect2 --device_map 2,3
# python sampling_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 2,3
# python sampling_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 2,3
# python sampling_vllm_dev.py --model_name mistral-8b-it --num_gen 5 --data_name webq --format reflect2 --device_map 2,3


# olmo2-7b-it
python greedy_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name triviaqa --format direct --device_map 0,1
python greedy_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name nq --format direct --device_map 0,1
python greedy_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name webq --format direct --device_map 0,1
python sampling_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name triviaqa --format direct --device_map 0,1
python sampling_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name nq --format direct --device_map 0,1
python sampling_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name webq --format direct --device_map 0,1

python greedy_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 0,1
python greedy_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
python greedy_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name webq --format reflect2 --device_map 0,1
python sampling_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 0,1
python sampling_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
python sampling_vllm_dev.py --model_name olmo2-7b-it --num_gen 5 --data_name webq --format reflect2 --device_map 0,1