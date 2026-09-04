# data loading code
import paths
import datasets, json, pickle
from tqdm import tqdm

def gsm8k_loading(format):
    new_dataset = []
    data = datasets.load_dataset("openai/gsm8k", "main")
    main_data = data['test']
    for idx, d in enumerate(main_data):
        q = d['question'] 
        q_id = f'gsm8k_{idx}'
        a = d['answer'].split("#### ")[-1]
        solution = d['answer'].split("#### ")[0]
        new_dataset.append({'question' : q,
                            'question_id' : q_id,
                            'answer' : a,
                            'solution' : solution})
    return new_dataset

def trivia_loading(format):
    new_dataset = []
    id_mem = set()
    def remove_dups(batch):
        if batch['question_id'][0] in id_mem:
            return {_:[] for _ in batch.keys()}
        id_mem.add(batch['question_id'][0])
        return batch
    
    data = datasets.load_dataset("trivia_qa", "rc", split="validation")
    data = data.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    data = data.remove_columns(["question_source", "entity_pages"]) # search_results는 남겨두자
        
    for d in data:
        q = d['question'] if d['question'].endswith("?") else d['question'] + "?"
        q_id = d['question_id']
        a = d['answer']['normalized_aliases']
        # 데이터 특성상 3-passage, 5-passage를 사용못함. 할려면 직접 dpr 모델로 돌려야됨
        ctx = f''
        for ct in d['search_results']['description']:
            ctx += ct
        new_dataset.append({'question': q,
                            'question_id' : q_id,
                            'answer' : a,
                            'contexts' : ctx})
            
    return new_dataset

def nq_loading(format):
    new_dataset = []
    # gold_passages_info 데이터는 일단 지우지 말자.
    path = f"{paths.DPR_DATA}/retriever_results/nq/single/test.json"
    with open(path, 'r') as f:
        data = json.load(f)
        
    for idx, d in enumerate(data):
        q = d['question'] if d['question'].endswith("?") else d['question'] + "?"
        q_id = 'nq_' + str(idx)
        a = d['answers']
        
        ctx = f''
        for i in range(3):
            ctx += '\n'
            ctx += d['ctxs'][i]['text']
        ctx = ctx if ctx.endswith(".") else ctx + "."
        new_dataset.append({'question' : q,
                            'question_id': q_id,
                            'answer' : a,
                            'contexts' : ctx})
            
    return new_dataset

def webq_loading(format):
    new_dataset = []
    path = f"{paths.DPR_DATA}/retriever/webq_mytest.pkl"
    with open(path, 'rb') as f:
        data = pickle.load(f)
    for idx, d in enumerate(data):
        q = d['question'] if d['question'].endswith("?") else d['question'] + "?"
        q_id = 'webq_' + str(idx)
        a = d['answers']
        
        ctx = f""
        for i in range(len(d['docs'])):
            ctx += '\n'
            ctx += d['docs'][i]
        ctx = ctx if ctx.endswith(".") else ctx + "."
        new_dataset.append({'question' : q,
                            'question_id' : q_id,
                            'answer' : a,
                            'contexts' : ctx})
        
    return new_dataset

def data_loading(data_name, format):
    if 'trivia' in data_name:
        loaded_data = trivia_loading(format)
        
    elif 'nq' in data_name:
        loaded_data = nq_loading(format)

    elif 'webq' in data_name:
        loaded_data = webq_loading(format)
        
    elif 'gsm8k' in data_name:
        loaded_data = gsm8k_loading(format)
        
    return loaded_data

def original_prompting(data_sample):
    question = data_sample
    prompted = f'''You are a smart generative AI.
Follow the guidelines below to verify your understanding of the user’s question and to generate your answer accordingly.

## Guidelines
Follow a one-step process:

### 1. [Answer]
- Based on your understanding, provide the most appropriate answer to the given question.
- Use a single word if possible.
- If a single word is ambiguous or insufficient, use a short noun phrase (2–3 words maximum).
- Never output a full sentence.


# Few-shot examples.

## Example 1
Q: From which country did Angola achieve independence in 1975?
[Answer]
Portugal

## Example 2
Q: Thomas Minton at Stoke on Trent created what in 1789?
[Answer]
The Willow Pattern

## Example 3
Q: In desert regions what is a ‘haboob’?
[Answer]
Thunderstorm

## Example 4
Q: In which London hotel was the Peach Melba invented in 1892?
[Answer]
SAVOY

## Example 5
Q: In 1954, who was the first Australian to win the British Open Golf Championship?
[Answer]
Kel Nagle


Now follow the guidelines above for the next user question.
Do NOT include any markdown headers (e.g., #, ##) in your output.
Only produce the four sections: [Answer].


Question: {question}'''
    return prompted

def reflection_prompting2(data_sample):
    question = data_sample
    inst_prompt = f'''You are a smart generative AI.
Follow the guidelines below to verify your understanding of the user’s question and to generate your answer accordingly.

## Guidelines
Follow a four-step process:

### 1. [Paraphrase]
- Generate three paraphrased versions of the given question, reflecting your understanding.
- Use different strategies: synonym replacement, sentence structure changes, tense/format variation, etc.

### 2. [Consistency]
- Based on the paraphrased versions, identify whether any of them differ in meaning from the original question.
- If one or more versions convey a different meaning, explicitly state which versions (e.g., “P2 and P3 differ in meaning from the Q”).
- If all paraphrased versions preserve the same meaning, explicitly state: “All P have the same meaning as the Q.”

### 3. [Understanding Score]
- Based on the [Consistency] results, assign a score from 0 to 100 that quantifies your overall understanding of the user’s question.
- Scoring rule:
 - All P correct: 90 to 100
 - 1/3 P incorrect: 60 to 80
 - 2/3 P incorrect: 30 to 50
 - All incorrect: 0

### 4. [Answer]
- Based on your verified understanding from the previous steps, provide the most appropriate answer to the given question.
- Use a single word if possible.
- If a single word is ambiguous or insufficient, use a short noun phrase (2–3 words maximum).
- Never output a full sentence.


# Few-shot examples.

## Example 1
Q: From which country did Angola achieve independence in 1975?
[Paraphrase]
P1: Which nation did Angola gain independence from in 1975?
P2: In 1975, Angola became independent from which country?
P3: Which country did Angola break away from in 1975?
[Consistency]
All paraphrased versions have the same meaning as the Q.
[Understanding Score]
Understanding score: 95
[Answer]
Portugal

## Example 2
Q: Thomas Minton at Stoke on Trent created what in 1789?
[Paraphrase]
P1: In 1789, what did Thomas Minton invent in Stoke on Trent?
P2: What was created by Thomas Minton in Stoke on Trent in the year 1789?
P3: What did Thomas Minton establish in 1789?
[Consistency]
P3 differs in meaning from the Q.
[Understanding Score]
Understanding score: 70
[Answer]
The Willow Pattern

## Example 3
Q: In desert regions what is a ‘haboob’?
[Paraphrase]
P1: What is a ‘haboob’ in desert areas?
P2: How does a 'haboob' form in desert regions?
P3: What types of 'haboobs' are there in desert regions?
[Consistency]
P2 and P3 differ in meaning from the Q.
[Understanding Score]
Understanding score: 30
[Answer]
Thunderstorm

## Example 4
Q: In which London hotel was the Peach Melba invented in 1892?
[Paraphrase]
P1: Which hotel in London was the birthplace of the Peach Melba in 1892?
P2: In 1892, at which London hotel was the Peach Melba created?
P3: What is the name of the London hotel where the Peach Melba was invented in 1892?
[Consistency]
All paraphrased versions have the same meaning as the Q.
[Understanding Score]
Understanding score: 90
[Answer]
SAVOY

## Example 5
Q: In 1954, who was the first Australian to win the British Open Golf Championship?
[Paraphrase]
Q1: What was the winning score of the player who recorded the first victory at the Open Championship in 1954?
Q2: In the 1954 British Open, who was the opponent in the final match against the first Australian?
Q3: After producing its first Australian champion in 1954, how many years passed before the next Australian won the Open Championship?
[Consistency]
P1, P2 and P3 differ in meaning from the Q.
[Understanding Score]
Understanding score: 0
[Answer]
Kel Nagle


Now follow the guidelines above for the next user question.
Do NOT include any markdown headers (e.g., #, ##) in your output.
Only produce the four sections: [Paraphrase], [Consistency], [Understanding Score], [Answer].

Q: {question}'''
    return inst_prompt

def reflection_prompting(data_sample):
    question = data_sample
    inst_prompt = f'''You are a smart generative AI.
Follow the guidelines below to verify your understanding of the user’s question and to generate your answer accordingly.

## Guidelines
Follow a three-step process:

### 1. [Paraphrase]
- Generate three paraphrased versions of the given question, reflecting your understanding.
- Use different strategies: synonym replacement, sentence structure changes, tense/format variation, etc.

### 2. [Consistency]
- Based on the paraphrased versions, identify whether any of them differ in meaning from the original question.
- If one or more versions convey a different meaning, explicitly state which versions (e.g., “P2 and P3 differ in meaning from the Q”).
- If all paraphrased versions preserve the same meaning, explicitly state: “All P have the same meaning as the Q.”

### 3. [Answer]
- Based on your verified understanding from the previous steps, provide the most appropriate answer to the given question.
- Use a single word if possible.
- If a single word is ambiguous or insufficient, use a short noun phrase (2–3 words maximum).
- Never output a full sentence.


# Few-shot examples.

## Example 1
Q: From which country did Angola achieve independence in 1975?
[Paraphrase]
P1: Which nation did Angola gain independence from in 1975?
P2: In 1975, Angola became independent from which country?
P3: Which country did Angola break away from in 1975?
[Consistency]
All paraphrased versions have the same meaning as the Q.
[Answer]
Portugal

## Example 2
Q: Thomas Minton at Stoke on Trent created what in 1789?
[Paraphrase]
P1: In 1789, what did Thomas Minton invent in Stoke on Trent?
P2: What was created by Thomas Minton in Stoke on Trent in the year 1789?
P3: What did Thomas Minton establish in 1789?
[Consistency]
P3 differs in meaning from the Q.
[Answer]
The Willow Pattern

## Example 3
Q: In desert regions what is a ‘haboob’?
[Paraphrase]
P1: What is a ‘haboob’ in desert areas?
P2: How does a 'haboob' form in desert regions?
P3: What types of 'haboobs' are there in desert regions?
[Consistency]
P2 and P3 differ in meaning from the Q.
[Answer]
Thunderstorm

## Example 4
Q: In which London hotel was the Peach Melba invented in 1892?
[Paraphrase]
P1: Which hotel in London was the birthplace of the Peach Melba in 1892?
P2: In 1892, at which London hotel was the Peach Melba created?
P3: What is the name of the London hotel where the Peach Melba was invented in 1892?
[Consistency]
All paraphrased versions have the same meaning as the Q.
[Answer]
SAVOY

## Example 5
Q: In 1954, who was the first Australian to win the British Open Golf Championship?
[Paraphrase]
Q1: What was the winning score of the player who recorded the first victory at the Open Championship in 1954?
Q2: In the 1954 British Open, who was the opponent in the final match against the first Australian?
Q3: After producing its first Australian champion in 1954, how many years passed before the next Australian won the Open Championship?
[Consistency]
P1, P2 and P3 differ in meaning from the Q.
[Answer]
Kel Nagle


Now follow the guidelines above for the next user question.
Do NOT include any markdown headers (e.g., #, ##) in your answer.
Only produce the three sections: [Paraphrase], [Consistency], [Answer].

Q: {question}'''
    return inst_prompt

def para_prompting(data_sample):
    question = data_sample
    inst_prompt = f'''You are a smart generative AI.
Follow the guidelines below to verify your understanding of the user’s question and to generate your answer accordingly.

## Guidelines
Follow a two-step process:

### 1. [Paraphrase]
- Generate three paraphrased versions of the given question, reflecting your understanding.
- Use different strategies: synonym replacement, sentence structure changes, tense/format variation, etc.

### 2. [Answer]
- Based on your verified understanding from the previous steps, provide the most appropriate answer to the given question.
- Use a single word if possible.
- If a single word is ambiguous or insufficient, use a short noun phrase (2–3 words maximum).
- Never output a full sentence.


# Few-shot examples.

## Example 1
Q: From which country did Angola achieve independence in 1975?
[Paraphrase]
P1: Which nation did Angola gain independence from in 1975?
P2: In 1975, Angola became independent from which country?
P3: Which country did Angola break away from in 1975?
[Answer]
Portugal

## Example 2
Q: Thomas Minton at Stoke on Trent created what in 1789?
[Paraphrase]
P1: In 1789, what did Thomas Minton invent in Stoke on Trent?
P2: What was created by Thomas Minton in Stoke on Trent in the year 1789?
P3: What did Thomas Minton establish in 1789?
[Answer]
The Willow Pattern

## Example 3
Q: In desert regions what is a ‘haboob’?
[Paraphrase]
P1: What is a ‘haboob’ in desert areas?
P2: How does a 'haboob' form in desert regions?
P3: What types of 'haboobs' are there in desert regions?
[Answer]
Thunderstorm

## Example 4
Q: In which London hotel was the Peach Melba invented in 1892?
[Paraphrase]
P1: Which hotel in London was the birthplace of the Peach Melba in 1892?
P2: In 1892, at which London hotel was the Peach Melba created?
P3: What is the name of the London hotel where the Peach Melba was invented in 1892?
[Answer]
SAVOY

## Example 5
Q: In 1954, who was the first Australian to win the British Open Golf Championship?
[Paraphrase]
Q1: What was the winning score of the player who recorded the first victory at the Open Championship in 1954?
Q2: In the 1954 British Open, who was the opponent in the final match against the first Australian?
Q3: After producing its first Australian champion in 1954, how many years passed before the next Australian won the Open Championship?
[Answer]
Kel Nagle


Now follow the guidelines above for the next user question.
Do NOT include any markdown headers (e.g., #, ##) in your output.
Only produce the two sections: [Paraphrase], [Answer].

Q: {question}'''
    return inst_prompt


# for gsm8k
def cot_prompting(data_sample):
    question = data_sample
    inst_prompt = r'''You are a smart generative AI.
Follow the guidelines below to generate your answer.

Guidelines
Follow a two-step process:

1. [CoT]
- Reasoning step by step to solve given math problem.

2. [Answer]
- Never output a full sentence.


Few-shot examples.

Example 1
Q: Find the remainder when $7^{100}$ is divided by $13$.
[CoT]
Step 1. By Fermat’s Little Theorem, since $7$ and $13$ are coprime, we know $7^{12} \equiv 1 \pmod{13}$.
Step 2. We compute $100 \bmod 12 = 4$. So $7^{100} \equiv 7^4 \pmod{13}$.
Step 3. Now $7^2 = 49 \equiv 10 \pmod{13}$. Then $7^4 \equiv 10^2 = 100 \equiv 9 \pmod{13}$.
[Answer]
9

Example 2
Q: Simplify $\frac{x^4 - 1}{x^2 - 1}$.
[CoT]
Step 1. Factor the numerator: $x^4 - 1 = (x^2 - 1)(x^2 + 1)$.
Step 2. Cancel $(x^2 - 1)$ from numerator and denominator.
Step 3. We get $x^2 + 1$.
[Answer]
x^2 + 1

Example 3
Q: How many integer solutions are there to $a + b + c = 10$ where $a, b, c \geq 0$?
[CoT]
Step 1. This is a stars-and-bars problem. The number of nonnegative integer solutions is $\binom{10+3-1}{3-1} = \binom{12}{2}$.
Step 2. $\binom{12}{2} = 66$.
[Answer]
66

Example 4
Q: Evaluate $\lim_{n \to \infty} \left( 1 + \frac{1}{n} \right)^n$.
[CoT]
Step 1. This is the standard limit definition of Euler’s number.
Step 2. As $n$ grows, $\left( 1 + \frac{1}{n} \right)^n \to e$.
[Answer]
e

Example 5
Q: Find $\gcd(252, 198)$.
[CoT]
Step 1. Apply the Euclidean Algorithm.
Step 2. $252 \div 198 = 1$ remainder $54$.
Step 3. $198 \div 54 = 3$ remainder $36$.
Step 4. $54 \div 36 = 1$ remainder $18$.
Step 5. $36 \div 18 = 2$ remainder $0$.
Step 6. So $\gcd(252, 198) = 18$.
[Answer]
18


Now follow the guidelines above for the next user question.
Do NOT include any markdown headers (e.g., *, **) in your output.
Only produce the two sections: [CoT], [Answer].

Q: '''
    my_prompt = inst_prompt + question
    return my_prompt


def cot_ref_prompting(data_sample):
    question = data_sample
    inst_prompt = r'''You are a smart generative AI.
Follow the guidelines below to verify your understanding of the user’s question and to generate your answer accordingly.

Guidelines
Follow a five-step process:

1. [Paraphrase]
- Generate three paraphrased versions of the given question, reflecting your understanding.
- Use different strategies: synonym replacement, sentence structure changes, tense/format variation, etc.

2. [Consistency]
- Based on the paraphrased versions, identify whether any of them differ in meaning from the original question.
- If one or more versions convey a different meaning, explicitly state which versions (e.g., “P2 and P3 differ in meaning from the Q”).
- If all paraphrased versions preserve the same meaning, explicitly state: “All P have the same meaning as the Q.”

3. [Understanding Score]
- Based on the [Consistency] results, assign a score from 0 to 100 that quantifies your overall understanding of the user’s question.
- Scoring rule:
 - All P correct: 90 to 100
 - 1/3 P incorrect: 60 to 80
 - 2/3 P incorrect: 30 to 50
 - All incorrect: 0

4. [CoT]
- Reasoning step by step to solve given math problem.

5. [Answer]
- Never output a full sentence.


Few-shot examples.

Example 1
Q: Find the remainder when $7^{100}$ is divided by $13$.
[Paraphrase]
P1: Determine the remainder obtained when dividing $7^{100}$ by 13.
P2: What is the remainder when $7^{100}$ is divided by 13?
P3: Find the remainder obtained when $7^{100}$ is divided by 13.
[Consistency]
All paraphrased versions have the same meaning as the Q.
[Understanding Score]
Understanding score: 100
[CoT]
Step 1. By Fermat’s Little Theorem, since $7$ and $13$ are coprime, we know $7^{12} \equiv 1 \pmod{13}$.
Step 2. We compute $100 \bmod 12 = 4$. So $7^{100} \equiv 7^4 \pmod{13}$.
Step 3. Now $7^2 = 49 \equiv 10 \pmod{13}$. Then $7^4 \equiv 10^2 = 100 \equiv 9 \pmod{13}$.
[Answer]
9

Example 2
Q: Simplify $\frac{x^4 - 1}{x^2 - 1}$.
[Paraphrase]
P1: Reduce the expression $\frac{x^4 - 1}{x^2 - 1}$
P2: Perform the simplification of the expression $\frac{x^4 - 1}{x^2 - 1}$.
P3: How can the fraction $\frac{x^4 - 1}{x^2 - 1}$ be simplified?
[Consistency]
All paraphrased versions have the same meaning as the Q.
[Understanding Score]
Understanding score: 95
[CoT]
Step 1. Factor the numerator: $x^4 - 1 = (x^2 - 1)(x^2 + 1)$.
Step 2. Cancel $(x^2 - 1)$ from numerator and denominator.
Step 3. We get $x^2 + 1$.
[Answer]
x^2 + 1

Example 3
Q: How many integer solutions are there to $a + b + c = 10$ where $a, b, c \geq 0$?
[Paraphrase]
P1: Find the number of non-negative integer triples (a,b,c) that satisfy the equation $a + b + c = 10$.
P2: If a,b, and c are non-negative integers such that their sum is 10, how many different sets of values can they take?
P3: How many positive integer solutions are there to $a + b + c = 10$?
[Consistency]
P3 differs in meaning from the Q.
[Understanding Score]
Understanding score: 70
[CoT]
Step 1. This is a stars-and-bars problem. The number of nonnegative integer solutions is $\binom{10+3-1}{3-1} = \binom{12}{2}$.
Step 2. $\binom{12}{2} = 66$.
[Answer]
66

Example 4
Q: Evaluate $\lim_{n \to \infty} \left( 1 + \frac{1}{n} \right)^n$.
[Paraphrase]
P1: Find the limit as n approaches infinity of the sequence $\left( 1 + \frac{1}{n} \right)^n$.
P2: Evaluate $\lim_{n \to 0^+} \left( 1 + \frac{1}{n} \right)^n$.
P3: Evaluate $\lim_{n \to \infty} \left( 1 + \frac{k}{n} \right)^n$ for a constant k > 0.
[Consistency]
P2 and P3 differ in meaning from the Q.
[Understanding Score]
Understanding score: 40
[CoT]
Step 1. This is the standard limit definition of Euler’s number.
Step 2. As $n$ grows, $\left( 1 + \frac{1}{n} \right)^n \to e$.
[Answer]
e

Example 5
Q: Find $\gcd(252, 198)$.
[Paraphrase]
P1: Determine the greatest common divisor of 252 and 198.
P2: What is the greatest common factor (GCF) of 252 and 198?
P3: Compute $\gcd(198, 252)$.
[Consistency]
All paraphrased versions have the same meaning as the Q.
[Understanding Score]
Understanding score: 100
[CoT]
Step 1. Apply the Euclidean Algorithm.
Step 2. $252 \div 198 = 1$ remainder $54$.
Step 3. $198 \div 54 = 3$ remainder $36$.
Step 4. $54 \div 36 = 1$ remainder $18$.
Step 5. $36 \div 18 = 2$ remainder $0$.
Step 6. So $\gcd(252, 198) = 18$.
[Answer]
18


Now follow the guidelines above for the next user question.
Do NOT include any markdown headers (e.g., *, **) in your output.
Only produce the five sections: [Paraphrase], [Consistency], [Understanding Score], [CoT], [Answer].

Q: '''
    my_prompt = inst_prompt + question
    return my_prompt

def prompting_data_generation(dataset, tokenizer, format):
    if 'direct' == format:
        prompt_list = [original_prompting(data['question']) for data in tqdm(dataset)]      
    elif 'reflect2' == format:
        prompt_list = [reflection_prompting2(data['question']) for data in tqdm(dataset)]
    elif 'para-under' == format:
        prompt_list = [reflection_prompting(data['question']) for data in tqdm(dataset)]
    elif 'paraphrase' == format:
        prompt_list = [para_prompting(data['question']) for data in tqdm(dataset)]
    elif 'rec' == format:
        prompt_list = [recitation_prompting(data['question']) for data in tqdm(dataset)]
    elif 'rag' == format:
        prompt_list = [rag_prompting(data['question'], data['contexts']) for data in tqdm(dataset)]
    my_data_list = []
    for idx, data in enumerate(dataset):
        inst_prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt_list[idx]}], tokenize=False, add_generation_prompt=True)

        if 'rag' in format:
            my_data_dict = {'question_id' : data['question_id'],
                            'question' : data['question'],
                            'answer' : data['answer'],
                            'formatted_input': inst_prompt,
                            'contexts' : data['contexts']}
            
        else:
            my_data_dict = {'question_id' : data['question_id'],
                            'question' : data['question'],
                            'answer' : data['answer'],
                            'formatted_input': inst_prompt,}
            
        my_data_list.append(my_data_dict)        
    
    return my_data_list

def recitation_prompting(question_data):
    inst_prompt = f'''Return answer to the given question after recitation. You must strictly follow the format below by using the symbols "[Recitation]", "Step 1.", "Step 2.", "Step 3.", "Step 4." and "[Answer]".

Q: given question
[Recitation]
Step 1. Grasping the core of the given question.
Step 2. Establishing a strategy for generating knowledge related to the given question.
Step 3. Generate knowledge related to the given question.
Step 4. Find answer to the given question.
[Answer]
answer

Here are examples.

Q: From which country did Angola achieve independence in 1975?
[Recitation]
Step 1. Key point of the question: The country from which Angola gained independence in 1975
Step 2. Generate independence information of Angola in 1975.
Step 3. The Angolan War of Independence was a war of independence fought between the Angolan nationalist forces of the MPLA, UNITA and FNLA, and Portugal.
Step 4. Angola achieved independence from Portugal in 1975.
[Answer]
Portugal

Q: Who plays andy's brother in the office?
[Recitation]
Step 1. Key point of the question: The person who plays andy's brother in the office.
Step 2. Generate information of the office and andy's brother in the office.
Step 3. "Garden Party" is the fourth episode of the eighth season of the American comedy television series The Office. The Office has cast singer Josh Groban as Andy Bernard’s brother, a day after revealing the identity.
Step 4. Josh Groban plays andy'brother in the office.
[Answer]
Josh Groban

Q: What is the distance between bases on a little league baseball field?
[Recitation]
Step 1. Key point of the question: The distance between bases on a little league baseball field.
Step 2. Generate information of The distance between bases on a little league baseball field.
Step 3. Generally, the distance between base paths on fields for 12-year-olds and below in baseball and in all divisions of softball is 60 feet.
Step 4. 60 feet is the distance between bases on a little league baseball field.
[Answer]
60 feet

Q: What is the capital of Nova Scotia (East Canada)?
[Recitation]
Step 1. Key point of the question: the capital of Nova Scotia(East Canada)
Step 2. Generate information of the capital related to Nova Scotia.
Step 3. Halifax is the capital of the province of Nova Scotia , Canada. The largest urban area in the Atlantic provinces.
Step 4. Halifax is the capital of Nova Scotia
[Answer]
Halifax

Q: In the human body, what does melanin determine?
[Recitation]
Step 1. Key point of the question: melanin determines this in the human body.
Step 2. Generate information of the effect of melanin in the human body.
Step 3. Melanin is responsible for the color of your skin. The more melanin produced in the skin, the darker the skin tone. It comes in several forms, including eumelanin (which is brown or black) and pheomelanin (which is yellow or red).
Step 4. Skin colour is determined by melanin
[Answer]
Skin colour


Now, return answer to the given question after recitation.

Q: {question_data}'''
    return inst_prompt

def rag_prompting(question_data, context_data):
    inst_prompt = f'''Following the guideline, you need to return the answer to the given question by referring to the context using the symbols '[Answer]', '[Judge]'.
    
guideline
 - If an answer to a question is in the context, extract the chunk containing the answer, and return the answer with the symbol '[Answer]'.
 - If an answer to a question is not in the context, first return 'The context does not contain the answer.' Then, return the answer to the question with the symbol '[Answer]'.
 - The response format you need to generate is as follows:  
[Judge]
judge
[Answer]
answer
 
Here are the examples of returning an answer to the question.

Q: Who sings does he love me with reba?
Context:
Does He Love You "Does He Love You" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba\'s album "Greatest Hits Volume Two". It is one of country music\'s several songs about a love triangle. "Does He Love You" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members.
leave the canteen next to them standing. This development process, driven by the need to bomb in unsighted conditions, meant that by the end of World War II, unguided RAF bombs could be predictably delivered within 25 yards of a target from 15,000 feet height, and precisely on it from low level. For the U.S. Army Air Forces, daylight bombing was normal based upon box formations for defence from fighters. Bombing was coordinated through a lead aircraft but although still nominally precision bombing (as opposed to the area bombing carried out by RAF Bomber Command) the result of bombing from
Hall Airport Hall Airport is a privately owned, public use airport located six nautical miles (11 km) northwest of the central business district of Kaufman, a city in Kaufman County, Texas, United States. Hall Airport covers an area of 27 acres (11 ha) at an elevation of 440 feet (134 m) above mean sea level. It has one runway designated 17/35 with a turf surface measuring 2,585 by 40 feet (788 x 12 m). For the 12-month period ending May 23, 2007, the airport had 201 general aviation aircraft operations, an average of 16 per month. At that time there

[Judge]
The answer is in the chunk '"Does He Love You" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis'.
[Answer]
Linda Davis


Q: Who was the creator of victoria 's secret?
Context:
local "half-breed" man of white and Eskimo descent. In a typical James Michener fashion, the final chapter is an interaction between various characters in preceding chapter or their descendants. Alaska is in the process of applying for statehood. Missy remains on the side advocating for statehood, while Tom Venn petitioned to keep Alaska a territory and under Seattle business control. In the end President Dwight D. Eisenhower signs the Alaska Statehood Act, making Alaska the 49th state of the Union. Michener invents characters and places although he also uses factual people or places in fictional events. Throughout the novel are
than to break off and declare independence. He is also instrumental in the selection of then-Colonel George Washington as the new head of the Continental Army. However, in his zeal for immediate action, he manages to alienate many of the other founding fathers, going so far as to insult John Dickinson, who is for conciliation to the Crown, implying that the man suffers from a religiously based moral cowardice. Later, Benjamin Franklin quietly chastens Adams, saying it is "perfectly acceptable to insult a man in private. He may even thank you for it afterwards. But when you do it in
the approximate location of Forty Fort. In the years following the Revolutionary War, Forty Fort became home to both the Nathan Denison House (built around 1790) and the Forty Fort Meetinghouse (built in 1806–08), which is located in the borough's cemetery. Forty Fort was officially incorporated as a borough in 1887. The borough later became home to the Lower School of the Wyoming Seminary and a portion of the southern end of the Wilkes-Barre Wyoming Valley Airport. In June 1972, Hurricane Agnes caused the Susquehanna River to overflow its banks. In Forty Fort, a portion of the levee protecting the

[Judge]
The answer is not in context. in my thought, the answer is Roy Raymond
[Answer]
Roy Raymond

Now, Refer to the context and return an appropriate answer to the question using the symbols [Answer], [Judge:].

Q: {question_data}
Context: {context_data}'''
    return inst_prompt

def reform_generation(dataset, tokenizer, format, rag_data):
    if format=='rec':
        prompt_data = recitation_prompting(dataset['question']) # recitation prompting
    elif format=='direct':
        prompt_data = original_prompting(dataset['question']) # pure prompting
    elif format=='rag':
        q_id = dataset['question_id']
        indexes = [idx for idx, rag_data in enumerate(rag_data) if rag_data["question_id"] == q_id] # 출력값이 리스트라 indexes[0]해줘야됨
        prompt_data = rag_prompting(dataset['question'], rag_data[indexes[0]]['contexts']) # rag prompting
    
    inst_prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt_data}], tokenize=False, add_generation_prompt=True)
    
    return inst_prompt