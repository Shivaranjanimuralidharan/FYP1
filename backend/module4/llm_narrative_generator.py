from datetime import datetime

def generate_narrative(prompt_obj, llm_client, config):

    prompt = prompt_obj["prompt"]

    text = llm_client.generate(
        system_prompt="You are a factual data analyst.",
        user_prompt=prompt
    )

    return {
        "text": text,
        "meta": {}
    }

