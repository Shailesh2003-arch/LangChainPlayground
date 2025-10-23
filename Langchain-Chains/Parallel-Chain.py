# A parallel chain is a setup where you run multiple chains at the same time on the same input (or different slices of it) and collect all outputs together.

# Example - From user you will get one input query like a long text, your task is to generate notes and quiz from the same large text.
# will send the query to one model and it will generate notes for us, and the same time we will take a second model and say it to generate quiz.
# And then we will merge the output of these two models.


from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")



prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text\n {text}",
    input_variables=['text']
)


prompt2 = PromptTemplate(
    template="Generate 5 short question answer from the following text \n {text}",
    input_variables=['text']
)


prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into the following two into a single document \n notes- > {notes} and quiz -> {quiz}",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)

model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)


# parallel chain...
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})


merge_chain = prompt3 | model1 | parser


# final chain... (Parallel chain connected to sequntial)

chain = parallel_chain | merge_chain

text = """"
How does generative AI work?
Simply put, researchers feed AI models large training datasets including text, images, audio, or other data types. These models recognise sophisticated patterns and structures within the dataset and can then generate new content based on those patterns.

Given this relationship, the quality and breadth of the training data set significantly impact the AI’s output. For example, an AI trained in a wide variety of human languages can generate more natural and contextually appropriate text.

Several key concepts and technologies (or deep learning models) that enable generative AI tools to function:

Neural networks and deep learning
Neural Networks are the foundational elements of generative AI. These mimic the human brain’s ability to ‘learn’ from data through a process called deep learning. By simulating the way the human brain processes information, they help machines to recognise patterns and relationships within data.
Deep learning involves multiple layers of neural networks that can learn hierarchical representations. This enables the generation of complex and nuanced content across various data types like text, images, and audio.
Transformers
Transformers are advanced models that help to understand context in human language. This is called natural language processing.
Many first-time users of tools like ChatGPT are impressed by the advances in transformers, which enable these AI tools to generate coherent and contextually relevant text as responses to human queries at speed. (Note: the GPT in ChatGPT stands for generative pre-trained transformer.)
Variational AutoEncoders (VAEs)
VAEs are like artists who study a bunch of paintings and then create new artworks in the same style. They learn the essence of the data (what makes a Van Gogh recognisable as a Van Gogh, for example) and can generate new items that resemble the originals. The output will be similar but crucially not identical to the original dataset.
Generative Adversarial Networks (GANs)
Imagine two players in a game: a generator and a discriminator.
The generator creates fake images and new data points based on a dataset.
The discriminator tries to detect real or fake images by comparing them to authentic data.
Over time, the creator gets better at making realistic images to ‘fool’ the detector. Through this adversarial training process, GANs work to produce high-quality images and videos: ai-generated content that can potentially pass as genuine.
Diffusion models
These models start with random noise and gradually turn it into meaningful data, like shaping a lump of clay into a sculpture step by step. They refine the data in stages until it becomes a clear and coherent output. Diffusion models have shown exceptional performance in generating high-resolution images and are gaining attention as a viable alternative to GANs.
Reinforcement Learning
Think of training a dog by giving it treats for good behaviour. Similarly, AI models learn the best actions to take by receiving rewards or penalties, helping them make better decisions over time. In this way, AI experts and engineers can shape the desired performance of AI models, fine-tuning them to achieve evermore specific goals.
For example, AI engineers can use reinforcement learning to reward models for adhering to ethical guidelines when training them.
"""

result = chain.invoke({
    'text':text
})


print(result)

chain.get_graph().print_ascii()