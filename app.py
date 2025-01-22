from elasticsearch import Elasticsearch
from scripts.find import load_and_prepare_data, BasicSearch, AdvancedElasticSearch, VectorSearch
from scripts.model import build_prompt
from scripts.model import llm
import streamlit as st
import logging
import uuid
import time
import gc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
HOST_URL = "http://localhost:9200"

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

es = Elasticsearch(HOST_URL)
data = load_and_prepare_data()

# try:
#     logging.info("Trying VectorSearch")
#     # vs = VectorSearch(data=data, index_name="got_question_v",num_results=5, search_client=es)
#     # vs.start_vs()
#     bs = BasicSearch(data=data, num_results=5)
#     search_algo = bs.create_index()

# except Exception as e:
#     logging.info(f"Vector search failed, trying Elastic Search {e}" )
#     els = AdvancedElasticSearch(data=data, index_name="got_question_e", num_results=5, search_client=es)
#     els.start_elastic_search()

# finally:
#     logging.info("Using Minsearch instead")
#     bs = BasicSearch(data=data, num_results=5)
#     search_algo = bs.create_index()


bs = BasicSearch(data=data, num_results=5)
search_algo = bs.create_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def response_gen(streaming_response):
    for word in streaming_response.split():
        yield f"{word} "
        time.sleep(0.05)

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # if not AdvancedElasticSearch.connect | VectorSearch.connect:
        search_results = bs.basic_search(query=prompt)
        print(search_results[0])
        # search_results = vs.vector_search(query=prompt)

        prompt = build_prompt(query=prompt, search_results=search_results)
        # Simulate stream of response with milliseconds delay
        streaming_response = llm(prompt)

        full_response = st.write_stream(response_gen(streaming_response))
            # full_response += chunk
            # message_placeholder.markdown(f"{full_response}▌")

        # message_placeholder.markdown(full_response)
            # st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
