# GPT API 사용할 수 있는 코드 완성했습니다. API KEY와 prompt만 입력하면 바로 사용 가능합니다.

from openai import OpenAI

NLP_API_KEY = "API KEY 여기에 넣으면 됩니다"
client = OpenAI(api_key = NLP_API_KEY)


completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": """prompt를 여기에 넣으면 됩니다"""}
  ]
)

print(completion.choices[0].message.content)
