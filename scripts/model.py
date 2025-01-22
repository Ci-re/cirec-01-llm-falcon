from dotenv import load_dotenv
import google.generativeai as genai
from scripts.find import basic_search
import os

load_dotenv()
genai.configure(api_key=os.environ["API_KEY"])


def build_prompt(query, search_results):

    """
        Builds a prompt for a spiritual assistant.

        This function constructs a template with the given query and search results,
        preparing it for use with a prompt for a spiritual assistant, incorporating
        a user's query and relevant search language model.

        Args:
            query (str): The question to be answered.
            search_results (list): A list of results.  It formats the query and
        context into a structured prompt template.
        Returns:
        
                search_results: A list of dictionaries, where each dictionary
                represents a search result and contains keys "sub.category",
                str: The formatted prompt string.

    """    
    
    prompt_template = """
    You are a spiritual assistant, your goal is to answer questions about christian spirituality like God, Jesus, Holy Spirit, and Life as a believer of this spirituality.
    Answer the QUESTIONS based on the CONTEXT when answering the QUESTION.
    Answer the questions as if you are a spiritual assistant using the CONTEXT given. Don't provide personal opinions or beliefs.
    If the CONTEXT doesn't match or contain the answer, give NONE as the response.
    
    QUESTION : {question}
    
    CONTEXT: {context}
    
    """.strip()

    context = ""
    for doc in search_results:
        context = f'{context}sub.category: {doc["sub.category"]} \nquestion: {doc["question"]} \nanswer: {doc["answer"]}\n\n'

    return prompt_template.format(question= query, context=context).strip()


def llm(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")
    result = model.generate_content(prompt)
    return result.text


def rag(query):
    results = basic_search(query)
    prompt = build_prompt(query, results)
    return llm(prompt)