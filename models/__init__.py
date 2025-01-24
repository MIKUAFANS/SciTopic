from openai import OpenAI
from .scitopic import TextEmbedding, TextCluster, MultiheadAttentionFusion, EntropySample, LLMQuery, LLMQueryOnline, TripletDataset, TripletLoss, ModelFineTune
from .evalution import evaluation_scitopic

def llm_query_processor(content, model, client_info, prompt):
    client = OpenAI(**client_info)
    conversation = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content},
    ]
    completion = client.chat.completions.create(messages=conversation, model=model)

    generated_text = completion.choices[0].message.content
    client.close()

    return generated_text