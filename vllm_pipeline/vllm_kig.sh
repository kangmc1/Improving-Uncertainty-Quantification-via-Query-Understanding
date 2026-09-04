# single kig 실험

# 1. llama
# python gen_kig_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format rec --device_map 0,1
# python gen_kig_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name nq --format rag --device_map 0,1
# python gen_kig_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name triviaqa --format rec --device_map 0,1
# python gen_kig_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name triviaqa --format rag --device_map 0,1
# python gen_kig_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name webq --format rec --device_map 0,1
# python gen_kig_vllm.py --model_name llama3.1-8b-it --num_gen 5 --data_name webq --format rag --device_map 0,1

# 2. qwen
# python gen_kig_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format rec --device_map 0,1
# python gen_kig_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name nq --format rag --device_map 0,1
# python gen_kig_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name triviaqa --format rec --device_map 0,1
# python gen_kig_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name triviaqa --format rag --device_map 0,1
# python gen_kig_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name webq --format rec --device_map 0,1
# python gen_kig_vllm.py --model_name qwen2.5-7b-it --num_gen 5 --data_name webq --format rag --device_map 0,1

# 3. mistral
# python gen_kig_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format rec --device_map 0,1
# python gen_kig_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name nq --format rag --device_map 0,1
# python gen_kig_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name triviaqa --format rec --device_map 0,1
# python gen_kig_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name triviaqa --format rag --device_map 0,1
# python gen_kig_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name webq --format rec --device_map 0,1
# python gen_kig_vllm.py --model_name mistral-8b-it --num_gen 5 --data_name webq --format rag --device_map 0,1


# 뭔가 쎄한데;;
# 5. olmo2-7b-it
# python gen_kig_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name nq --format rec --device_map 0,1
python gen_kig_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name nq --format rag --device_map 0,1
# python gen_kig_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name triviaqa --format rec --device_map 0,1
python gen_kig_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name triviaqa --format rag --device_map 0,1
# python gen_kig_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name webq --format rec --device_map 0,1
python gen_kig_vllm.py --model_name olmo2-7b-it --num_gen 5 --data_name webq --format rag --device_map 0,1