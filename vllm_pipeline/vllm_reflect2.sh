# 수정된 reflect generation

python greedy_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
python sampling_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1

python greedy_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
python sampling_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1

python greedy_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1
python sampling_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format reflect2 --device_map 0,1


# 수정된 reflect evaluation
python greedy_eval_vllm.py --model_name llama3.1-8b-it --data_name nq --format reflect2 --device_map 0
python sampling_eval_vllm.py --model_name llama3.1-8b-it --data_name nq --format reflect2 --device_map 0

python greedy_eval_vllm.py --model_name qwen2.5-7b-it --data_name nq --format reflect2 --device_map 0
python sampling_eval_vllm.py --model_name qwen2.5-7b-it --data_name nq --format reflect2 --device_map 0

python greedy_eval_vllm.py --model_name mistral-8b-it --data_name nq --format reflect2 --device_map 0
python sampling_eval_vllm.py --model_name mistral-8b-it --data_name nq --format reflect2 --device_map 0


