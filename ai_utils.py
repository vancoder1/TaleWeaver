from groq import Groq

class AIClient:
    def __init__(self, api_key, model, system_prompt):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
    
    def generate_text(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        stream = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stop=None,
            stream=True
        )

        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response