# vllm으로 진행한 모든 결과에 evaluation을 만들고 저장

# python greedy_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name triviaqa --format direct --device_map 3
# python sampling_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name triviaqa --format direct --device_map 3
# python greedy_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name nq --format direct --device_map 3
# python sampling_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name nq --format direct --device_map 3
# python greedy_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name webq --format direct --device_map 3
# python sampling_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name webq --format direct --device_map 3

# python greedy_eval_vllm.py --split dev --model_name qwen2.5-7b-it --data_name triviaqa --format direct --device_map 3
# python sampling_eval_vllm.py --split dev --model_name qwen2.5-7b-it --data_name triviaqa --format direct --device_map 3
# python greedy_eval_vllm.py --split dev --model_name qwen2.5-7b-it --data_name nq --format direct --device_map 3
# python sampling_eval_vllm.py --split dev --model_name qwen2.5-7b-it --data_name nq --format direct --device_map 3
# python greedy_eval_vllm.py --split dev --model_name qwen2.5-7-bit --data_name webq --format direct --device_map 3
# python sampling_eval_vllm.py --split dev --model_name qwen2.5-7b-it --data_name webq --format direct --device_map 3

# python greedy_eval_vllm.py --split dev --model_name mistral-8b-it --data_name nq --format direct --device_map 3
# python sampling_eval_vllm.py --split dev --model_name mistral-8b-it --data_name nq --format direct --device_map 3
# python greedy_eval_vllm.py --split dev --model_name mistral-8b-it --data_name triviaqa --format direct --device_map 3
# python sampling_eval_vllm.py --split dev --model_name mistral-8b-it --data_name triviaqa --format direct --device_map 3
# python greedy_eval_vllm.py --split dev --model_name mistral-8b-it --data_name webq --format direct --device_map 3
# python sampling_eval_vllm.py --split dev --model_name mistral-8b-it --data_name webq --format direct --device_map 3


# python greedy_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name triviaqa --format reflect2 --device_map 3
# python sampling_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name triviaqa --format reflect2 --device_map 3
# python greedy_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name nq --format reflect2 --device_map 3
# python sampling_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name nq --format reflect2 --device_map 3
# python greedy_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name webq --format reflect2 --device_map 3
# python sampling_eval_vllm.py --split dev --model_name llama3.1-8b-it --data_name webq --format reflect2 --device_map 3

# python greedy_eval_vllm.py --split dev --model_name qwen2.5-7b-it --data_name triviaqa --format reflect2 --device_map 3
# python sampling_eval_vllm.py --split dev --model_name qwen2.5-7b-it --data_name triviaqa --format reflect2 --device_map 3
# python greedy_eval_vllm.py --split dev --model_name qwen2.5-7b-it --data_name nq --format reflect2 --device_map 3
# python sampling_eval_vllm.py --split dev --model_name qwen2.5-7b-it --data_name nq --format reflect2 --device_map 3
# python greedy_eval_vllm.py --split dev --model_name qwen2.5-7b-it --data_name webq --format reflect2 --device_map 3
# python sampling_eval_vllm.py --split dev --model_name qwen2.5-7b-it --data_name webq --format reflect2 --device_map 3

# python greedy_eval_vllm.py --split dev --model_name mistral-8b-it --data_name nq --format reflect2 --device_map 3
# python sampling_eval_vllm.py --split dev --model_name mistral-8b-it --data_name nq --format reflect2 --device_map 3
# python greedy_eval_vllm.py --split dev --model_name mistral-8b-it --data_name triviaqa --format reflect2 --device_map 3
# python sampling_eval_vllm.py --split dev --model_name mistral-8b-it --data_name triviaqa --format reflect2 --device_map 3
# python greedy_eval_vllm.py --split dev --model_name mistral-8b-it --data_name webq --format reflect2 --device_map 3
# python sampling_eval_vllm.py --split dev --model_name mistral-8b-it --data_name webq --format reflect2 --device_map 3


# olmo2-7b-it

# 이거 꼭 돌려야됨;;
python greedy_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name nq --format direct --device_map 1
python sampling_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name nq --format direct --device_map 1
python greedy_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name triviaqa --format direct --device_map 1
python sampling_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name triviaqa --format direct --device_map 1
python greedy_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name webq --format direct --device_map 1
python sampling_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name webq --format direct --device_map 1

# python greedy_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name nq --format reflect2 --device_map 1
# python sampling_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name nq --format reflect2 --device_map 1
# python greedy_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name triviaqa --format reflect2 --device_map 1
# python sampling_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name triviaqa --format reflect2 --device_map 1
# python greedy_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name webq --format reflect2 --device_map 1
# python sampling_eval_vllm.py --split dev --model_name olmo2-7b-it --data_name webq --format reflect2 --device_map 1