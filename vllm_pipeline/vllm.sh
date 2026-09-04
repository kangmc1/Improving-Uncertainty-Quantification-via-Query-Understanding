# reflect 2s가 더 뛰어난지 확인하는 실험

# 1. llama
# python greedy_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name triviaqa --format direct --device_map 0,1
# python greedy_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format direct --device_map 0,1
# python greedy_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name webq --format direct --device_map 0,1
# python sampling_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name triviaqa --format direct --device_map 0,1
# python sampling_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format direct --device_map 0,1
# python sampling_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name webq --format direct --device_map 0,1

# 2. qwen
# python greedy_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name triviaqa --format direct --device_map 0,1
# python greedy_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format direct --device_map 0,1
# python greedy_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name webq --format direct --device_map 0,1
# python sampling_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name triviaqa --format direct --device_map 0,1
# python sampling_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format direct --device_map 0,1
# python sampling_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name webq --format direct --device_map 0,1

# 3. mistral(정식명칭은 ministral)
# python greedy_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name triviaqa --format direct --device_map 0,1
# python greedy_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format direct --device_map 0,1
# python greedy_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name webq --format direct --device_map 0,1
# python sampling_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name triviaqa --format direct --device_map 0,1
# python sampling_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format direct --device_map 0,1
# python sampling_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name webq --format direct --device_map 0,1


# reflect generation

# 1. llama
# python greedy_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 0,1
# python greedy_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
# python greedy_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name webq --format reflect2 --device_map 0,1
# python sampling_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 0,1
# python sampling_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
# python sampling_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name webq --format reflect2 --device_map 0,1

# 2. qwen
# python greedy_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 0,1
# python greedy_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
# python greedy_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name webq --format reflect2 --device_map 0,1
# python sampling_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 0,1
# python sampling_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
# python sampling_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name webq --format reflect2 --device_map 0,1

# 3. mistral(정식명칭은 ministral)
# python greedy_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 0,1
# python greedy_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
# python greedy_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name webq --format reflect2 --device_map 0,1
# python sampling_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 0,1
# python sampling_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
# python sampling_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name webq --format reflect2 --device_map 0,1


python greedy_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name triviaqa --format direct --device_map 0,1
python greedy_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name nq --format direct --device_map 0,1
python greedy_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name webq --format direct --device_map 0,1
python sampling_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name triviaqa --format direct --device_map 0,1
python sampling_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name nq --format direct --device_map 0,1
python sampling_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name webq --format direct --device_map 0,1

python greedy_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 0,1
python greedy_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
python greedy_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name webq --format reflect2 --device_map 0,1
python sampling_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name triviaqa --format reflect2 --device_map 0,1
python sampling_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
python sampling_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name webq --format reflect2 --device_map 0,1