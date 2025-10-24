from langchain_classic.text_splitter import RecursiveCharacterTextSplitter


text = """
Generative AI is a type of artificial intelligence that creates new content, such as text, images, and code, by learning patterns from existing data. It uses deep learning models and algorithms to generate novel outputs that are statistically similar to the data it was trained on, often in response to user prompts. Examples include creating art from a text description or generating human-like text for a chatbot.
"""


# first configure the splitter with CharacterTextSplitter.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0

)


splitted_text = splitter.split_text(text)
print(splitted_text)

# Recursive-Character-Text-Splitter is more better than textSplitter, as it uses recursive approach to make chunking.
# And it does a very good work. As it always tries to keep words intact.