# This is a length-based Text-Splitter which splits the text/document based on the length of the chunk-size.

# split_text
# split_document

from langchain_classic.text_splitter import CharacterTextSplitter


text = """
Generative AI is a type of artificial intelligence that creates new content, such as text, images, and code, by learning patterns from existing data. It uses deep learning models and algorithms to generate novel outputs that are statistically similar to the data it was trained on, often in response to user prompts. Examples include creating art from a text description or generating human-like text for a chatbot.
"""


# first configure the splitter with CharacterTextSplitter.
splitter = CharacterTextSplitter(
    chunk_size=100,
    separator='',
    chunk_overlap=15

)

# Now here this approach - CharacterTextSplitter() splits the text exactly on the specified chunk-size. And thus it can easily loose context.
# To make ensure a little context is preserved we use the parameter - chunk_overlap.
# Here this chunk-overlap shows how many characters to preserve from the last chunk in the next continuing chunk so that the context is maintained.

splitted_text = splitter.split_text(text)
print(splitted_text)