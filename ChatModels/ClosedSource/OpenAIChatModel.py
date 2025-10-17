from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv

model = ChatOpenAI(model="gpt-4",temperature=0.2)

# temperature param controls the randomness of a language model's output.
# It affects how creative or deterministic the responses are.
# Lower values -> 0.0 - 0.3 More deterministic and predictable.
# Higher values -> 0.7 - 1.5 More random creative and diverse.


# max_completion tokens... 

# for restricting the token 

result = model.invoke("Who are you?")
print(result.content)
