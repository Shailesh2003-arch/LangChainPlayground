from langchain_community.tools import DuckDuckGoSearchRun


# DuckDuckGo is a search engine provided by langchain as a tool for common web-search task.
tool = DuckDuckGoSearchRun()

result = tool.invoke("Real Name of Shah Rukh Khan")
print(result)
print(type(result))


# There are way more tools in langchain that are provided for solving general-purpose needs, checkout them at: https://docs.langchain.com/oss/python/integrations/tools