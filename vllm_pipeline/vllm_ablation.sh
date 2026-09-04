# nq 데이터에 대해서만 수행하자.


# data generation

# python greedy_vllm.py --split ablation --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format paraphrase --device_map 0,1
python greedy_vllm.py --split ablation --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format para-under --device_map 0,1
# python greedy_vllm.py --split ablation --model_name mistral-8b-it --num_gen 5 --data_name nq --format paraphrase --device_map 0,1
python greedy_vllm.py --split ablation --model_name mistral-8b-it --num_gen 5 --data_name nq --format para-under --device_map 0,1
# python greedy_vllm.py --split ablation --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format paraphrase --device_map 0,1
python greedy_vllm.py --split ablation --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format para-under --device_map 0,1

# python sampling_vllm.py --split ablation --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format paraphrase --device_map 0,1
python sampling_vllm.py --split ablation --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format para-under --device_map 0,1
# python sampling_vllm.py --split ablation --model_name mistral-8b-it --num_gen 5 --data_name nq --format paraphrase --device_map 0,1
python sampling_vllm.py --split ablation --model_name mistral-8b-it --num_gen 5 --data_name nq --format para-under --device_map 0,1
# python sampling_vllm.py --split ablation --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format paraphrase --device_map 0,1
python sampling_vllm.py --split ablation --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format para-under --device_map 0,1

# evaluation generation(greedy)
# python greedy_eval_vllm.py --split ablation --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format paraphrase
python greedy_eval_vllm.py --split ablation --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format para-under
# python greedy_eval_vllm.py --split ablation --model_name mistral-8b-it --num_gen 5 --data_name nq --format paraphrase
python greedy_eval_vllm.py --split ablation --model_name mistral-8b-it --num_gen 5 --data_name nq --format para-under
# python greedy_eval_vllm.py --split ablation --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format paraphrase
python greedy_eval_vllm.py --split ablation --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format para-under

# evaluation generation(sampling)
# python sampling_eval_vllm.py --split ablation --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format paraphrase --device_map 0,1
python sampling_eval_vllm.py --split ablation --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format para-under --device_map 0,1
# python sampling_eval_vllm.py --split ablation --model_name mistral-8b-it --num_gen 5 --data_name nq --format paraphrase --device_map 0,1
python sampling_eval_vllm.py --split ablation --model_name mistral-8b-it --num_gen 5 --data_name nq --format para-under --device_map 0,1
# python sampling_eval_vllm.py --split ablation --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format paraphrase --device_map 0,1
python sampling_eval_vllm.py --split ablation --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format para-under --device_map 0,1
