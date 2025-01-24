import numpy as np
from tqdm import tqdm
import tqdm as tm
from vllm import LLM, SamplingParams
from openai import OpenAI
import joblib

def llm_query_processor(item, model, client_info, prompt):
    client = OpenAI(**client_info)
    conversation = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": item["prompt"]},
    ]
    completion = client.chat.completions.create(messages=conversation, model=model)

    generated_text = completion.choices[0].message.content
    item["llm_response"] = generated_text
    # print(f"Generated text: {generated_text}")
    client.close()

    return item

def generate_query_prompt(prompt_data):

    prompt = f"""
        Query:
        Title: {prompt_data["title"]}
        Abstract: {prompt_data["abstraction"]}

        Choose the option most similar in topic to the Query:

        Choice 1:
        Title: {prompt_data["choice_title1"]}
        Abstract: {prompt_data["choice_abstraction1"]}

        Choice 2:
        Title: {prompt_data["choice_title2"]}
        Abstract: {prompt_data["choice_abstraction2"]}

        Respond only with 'Choice 1' or 'Choice 2' based on the closest topic similarity. Do not provide any explanation. You must choice 'Choice 1', 'Choice 2' or 'Neither'.
        """
    
    return prompt

def compute_entropy(cluster_centers, embedding):
    cluster_entropy = []
    closest = []
    alpha = 1
    epsilon = 0.5
    K_closest = max(int(epsilon * len(cluster_centers)), 2)
    for point in tqdm(embedding):
        distance = np.linalg.norm(cluster_centers - point, axis=1)
        every_point_prob = (1 + distance / alpha) ** (- (alpha + 1) / 2)
        prob = every_point_prob / np.sum(every_point_prob)
        argmax = np.argsort(prob)[::-1][:K_closest]
        K_closest_prob = prob[argmax]
        entropy = -np.sum(K_closest_prob * np.log(K_closest_prob))
        cluster_entropy.append(np.array(entropy))
        closest.append(np.array(argmax))

    return np.array(cluster_entropy), np.array(closest)

class EntropySample:
    def __init__(self, cluster_centers, cluster_labels, embedding, cluster_text, gamma_low=0.0, gamma_high=0.2):
        self.cluster_centers = cluster_centers
        self.cluster_labels = cluster_labels
        self.embedding = embedding
        self.cluster_text = cluster_text
        self.gamma_low = gamma_low
        self.gamma_high = gamma_high

        self.cluster_entropy, self.closest_center = compute_entropy(self.cluster_centers, self.embedding)
        self.sorted_cluster_entropy = np.argsort(self.cluster_entropy)[::-1][int(self.cluster_entropy.shape[0] * self.gamma_low): int(self.cluster_entropy.shape[0] * self.gamma_high)]
        np.random.shuffle(self.sorted_cluster_entropy)

    def generate_triplets(self):
        self.triplets = []
        for idx in tqdm(self.sorted_cluster_entropy):
            while True:
                cluster_label1, cluster_label2 = np.random.choice(self.closest_center[idx], 2, replace=False)
                choice1 = np.random.choice(self.cluster_text[str(self.cluster_labels[cluster_label1])], 1)
                choice2 = np.random.choice(self.cluster_text[str(self.cluster_labels[cluster_label2])], 1)

                if (idx, choice1[0], choice2[0]) not in self.triplets and idx != choice1[0] and idx != choice2[0]:
                    self.triplets.append((idx, choice1[0], choice2[0]))
                    break

        return self.triplets
    
    def generate_query(self, title : list, abstract : list):
        self.llm_query = []
        for idx, choice1, choice2 in self.triplets:
            is_random = False
            if np.random.rand() > 0.5:
                prompt_data = {
                    "title": title[idx],
                    "abstraction": abstract[idx],
                    "choice_title1": title[choice1],
                    "choice_abstraction1": abstract[choice1],
                    "choice_label1": self.cluster_labels[choice1],
                    "choice_title2": title[choice2],
                    "choice_abstraction2": abstract[choice2],
                    "choice_label2": self.cluster_labels[choice2]
                }

                is_random = True
            else:
                prompt_data = {
                    "title": title[idx],
                    "abstraction": abstract[idx],
                    "choice_title1": title[choice2],
                    "choice_abstraction1": abstract[choice2],
                    "choice_label1": self.cluster_labels[choice2],
                    "choice_title2": title[choice1],
                    "choice_abstraction2": abstract[choice1],
                    "choice_label2": self.cluster_labels[choice1]
                }

            if self.cluster_labels[idx] == self.cluster_labels[choice1] and self.cluster_labels[idx] == self.cluster_labels[choice2]:
                output = 0
            elif self.cluster_labels[idx] == self.cluster_labels[choice1] and self.cluster_labels[idx] != self.cluster_labels[choice2]:
                output = 1 if is_random else 2
            elif self.cluster_labels[idx] != self.cluster_labels[choice1] and self.cluster_labels[idx] == self.cluster_labels[choice2]:
                output = 2 if is_random else 1
            else:
                output = -1

            # prompt, input_data = generate_query(prompt_data)

            self.llm_query.append({
                "prompt": generate_query_prompt(prompt_data),
                # "input_data": input_data,
                "idx": int(idx),
                "choice1_idx": int(choice1),
                "choice2_idx": int(choice2),
                "output": int(output)
            })

        max_len = []
        for query in self.llm_query:
            max_len.append(max(len(query["prompt"]), 0))

        return self.llm_query, np.array(max_len)

    def __call__(self, title : list, abstract : list):
        if title is None:
            title = [""] * len(self.cluster_labels)
        if abstract is None:
            abstract = [""] * len(self.cluster_labels)

        assert len(title) == len(abstract) >= len(self.cluster_labels), f"The length of title {len(title)}, abstract {len(abstract)} and cluster_labels {len(self.cluster_labels)} must be the same."
        
        self.generate_triplets()
        return self.generate_query(title, abstract)


class LLMQuery:
    def __init__(self, model_path : str, max_model_len : int = 9216, tensor_parallel_size : int = 1, max_seq_len_to_capture : int = 9216) -> None:
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len, max_seq_len_to_capture=max_seq_len_to_capture)
        self.sampling_params = SamplingParams(temperature=0., stop=["</s>"], max_tokens=1024)
        self.prompt = """
            Select the title and abstract of paper that better corresponds with the Query in terms of scientific topic. There only threee response: 'Choice 1', 'Choice 2' or 'Neither'.
        """

    @staticmethod
    def print_outputs(outputs):
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Generated text: {generated_text}")
        print("-" * 10)

        return generated_text
        
    def __call__(self, llm_query):
        llm_query_results = []
        for item in tqdm(llm_query):
            conversation = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": item["prompt"]},
            ]
            llm_response = self.llm.chat(conversation, self.sampling_params, use_tqdm=False)
            generated_text = self.print_outputs(llm_response)
            item["llm_response"] = generated_text
            llm_query_results.append(item)

        return llm_query_results

class LLMQueryOnline:
    def __init__(self, api_key : str = None, model : str = "Meta-Llama-3.1-70B-Instruct", base_url : str = "http://10.0.82.200:8000/v1", prompt : str = None) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.client_info = {
            "api_key": api_key,
            "base_url": base_url
        }
        self.model = model
        # self.prompt = """
        #     Select the title and abstract of paper that better corresponds with the Query in terms of scientific topic. You must choice 'Choice 1' or 'Choice 2'. Only answer 'Choice 1' or 'Choice 2'.
        # """
        if prompt is None:
            self.prompt = """
                Select the title and abstract of paper that better corresponds with the Query in terms of scientific topic. You must choice 'Choice 1', 'Choice 2' or Nither. Only answer 'Choice 1', 'Choice 2' or Nither.
            """
        else:
            self.prompt = prompt


    def __call__(self, llm_query):
        llm_query_results = joblib.Parallel(n_jobs=8)(joblib.delayed(llm_query_processor)(item, self.model, self.client_info, self.prompt) for item in tm.tqdm(llm_query))

        return llm_query_results