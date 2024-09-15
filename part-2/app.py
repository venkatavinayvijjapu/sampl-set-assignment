import streamlit as st
import os
from dotenv import load_dotenv
from selenium import webdriver
from bs4 import BeautifulSoup
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
from typing import List
from uuid import uuid4
import textwrap
import pinecone
from tqdm.auto import tqdm
import time
# import openai
from langchain.llms import OpenAIChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
# from langchain_openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone import ServerlessSpec

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

api_key = 'gpt-3.5-turbo'
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'
client = OpenAI(api_key=OPENAI_API_KEY)

# st.title("Question and answering using URL's")
if 'pc' not in st.session_state:
    st.session_state['pc'] = Pinecone(api_key=PINECONE_API_KEY)
    # st.session_state['spec'] = ServerlessSpec(cloud="aws", region="us-east-1")
    
with st.sidebar:
    url=st.text_input("Enter your URL")
    if url and 'url' not in st.session_state:
        driver = webdriver.Chrome()  # You need to have the ChromeDriver installed and in your PATH
        driver.get(url)

            # Get the page source after JavaScript execution
        page_source = driver.page_source

            # Parse the page source with BeautifulSoup
        soup = BeautifulSoup(page_source, "html.parser")

            # Find and print the textual content
        textual_content = soup.get_text()
        print(textual_content)
            # Close the browser
        driver.quit()
        st.session_state['url']=url
    pinecone_name=st.text_area("Enter your pinecone APP name")
    if pinecone_name and 'pinecone' not in st.session_state:
        
        index_name=pinecone_name
        if index_name not in st.session_state['pc'].list_indexes():
        # if does not exist, create index

            st.session_state['pc'].create_index(
            name=index_name,
            dimension=1536, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
            # connect to index
        st.session_state['index'] = st.session_state['pc'].Index(index_name)
            # view index stats
        st.session_state['index'].describe_index_stats()
        print('success')
        st.session_state['pinecone']=pinecone_name
    



if (url is None or url==''):
    st.info("Provide your URL")
elif (pinecone is None or pinecone==''):
    st.info("Provide your pinecone app name")
else:
    def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
        return client.embeddings.create(input=text_or_tokens, model=model).data[0].embedding

    def chunk_text(text: str, max_chunk_size: int, overlap_size: int) -> List[str]:
        """Helper function to chunk a text into overlapping chunks of specified size."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            chunks.append(text[start:end])
            start += max_chunk_size - overlap_size
        return chunks

    def transform_record(record: dict) -> List[dict]:
        """Transform a single record as described in the prompt."""
        max_chunk_size = 500
        overlap_size = 100
        chunks = chunk_text(record, max_chunk_size, overlap_size)
        transformed_records = []
        recordId = str(uuid4())
        for i, chunk in enumerate(chunks):
            chunk_id = f"{recordId}-{i+1}"
            response=get_embedding(chunk)
            transformed_records.append({
                'chunk_id': chunk_id,
                'chunk_parent_id': recordId,
                'chunk_text': chunk,
                'vector' : response
                # embeddings.append(response['data'][0]['embedding'])
                #'sparse_values': splade(chunk)
            })
        return transformed_records
    if 'chunked_data' not in st.session_state:
        chunked_data = []
        chunk_array = transform_record(textual_content)
        for chunk in chunk_array:
            chunked_data.append(chunk)
        st.session_state['chunked_data']=chunked_data
        print(st.session_state['chunked_data'])
    def prepare_entries_for_pinecone(entries):
        """
        Prepares an array of entries for upsert to Pinecone.
        Each entry should have a 'vector' field containing a list of floats.
        """
        vectors = []
        for entry in entries:
            vector = entry['vector']
            id = entry.get('chunk_id', '')
            metadata = entry.get('metadata', {'chunk_id': entry.get('chunk_id', ''),'parent_id': entry.get('chunk_parent_id', ''), 'chunk_text': entry.get('chunk_text', '')})
            values = [v for v in vector]
            # sparse_values = entry['sparse_values']
            #vectors.append({'id': id, 'metadata': metadata, 'values': values, 'sparse_values': sparse_values})
            vectors.append({'id': id, 'metadata': metadata, 'values': values})
        return {'vectors': vectors, 'namespace': ''}
    if 'vectors' not in st.session_state:
        st.session_state['vectors'] = prepare_entries_for_pinecone(st.session_state['chunked_data'])
        vectors=st.session_state['vectors']
        batch_size = 32
        for i in tqdm(range(0, len(vectors['vectors']), batch_size)):
            ids_batch = [id['id'] for id in vectors['vectors'][i:i+batch_size]]
            embeds = [id['values'] for id in vectors['vectors'][i:i+batch_size]]
            meta = [id['metadata'] for id in vectors['vectors'][i:i+batch_size]]
            # sparse_values = [id['sparse_values'] for id in vectors['vectors'][i:i+batch_size]]
            upserts = []
            # loop through the data and create dictionaries for uploading documents to pinecone index
            # for _id, sparse, dense, meta in zip(ids_batch, sparse_values, embeds, meta):
            for _id,dense, meta in zip(ids_batch, embeds, meta):
                upserts.append({
                    'id': _id,
                    # 'sparse_values': sparse,
                    'values': dense,
                    'metadata': meta
                })
            # upload the documents to the new hybrid index
            st.session_state['index'].upsert(upserts)
    limit = 8000
    def retrieve(query):
        res = client.embeddings.create(
            input=[query],
            model='text-embedding-ada-002'
        ).data[0].embedding

        # retrieve from Pinecone
        xq = res

        # get relevant contexts
        res = st.session_state['index'].query(vector=[xq], top_k=5, include_metadata=True)
        contexts = [
            x['metadata']['chunk_text'] for x in res['matches']
        ]

        # build our prompt with the retrieved contexts included
        prompt_start = (
            "Answer the question based on the context below. If you cannot answer based on the context or general knowledge about Wells Fargo, truthfully answer that you don't know.\n\n" +
            "Context:\n"
        )
        prompt_end = f"\n\nQuestion: {query}\nAnswer:"
        
        # Initialize prompt in case no context is retrieved or the loop does not run
        prompt = prompt_start + prompt_end

        # append contexts until hitting the limit
        for i in range(1, len(contexts) + 1):
            if len("\n\n---\n\n".join(contexts[:i])) >= limit:
                prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts[:i-1]) +
                    prompt_end
                )
                break
            elif i == len(contexts):
                prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts) +
                    prompt_end
                )
        
        return prompt

    # def retrieve(query):
    #     res = client.embeddings.create(
    #         input=[query],
    #         model='text-embedding-ada-002'
    #     ).data[0].embedding

    #     # retrieve from Pinecone
    #     xq = res
    #     #sq = splade(query)


    #     # get relevant contexts
    #     #res = index.query(xq, top_k=5, include_metadata=True, sparse_vector=sq)
    #     res = st.session_state['index'].query(vector=[xq], top_k=5, include_metadata=True)
    #     contexts = [
    #         x['metadata']['chunk_text'] for x in res['matches']
    #     ]

    #     # build our prompt with the retrieved contexts included
    #     prompt_start = (
    #         "Answer the question based on the context below. If you cannot answer based on the context or general knowledge about Wells Fargo, truthfully answer that you don't know.\n\n"+
    #         "Context:\n"
    #     )
    #     prompt_end = (
    #         f"\n\nQuestion: {query}\nAnswer:"
    #     )
    #     # append contexts until hitting limit
    #     for i in range(1, len(contexts)):
    #         if len("\n\n---\n\n".join(contexts[:i])) >= limit:
    #             prompt = (
    #                 prompt_start +
    #                 "\n\n---\n\n".join(contexts[:i-1]) +
    #                 prompt_end
    #             )
    #             break
    #         elif i == len(contexts)-1:
    #             prompt = (
    #                 prompt_start +
    #                 "\n\n---\n\n".join(contexts) +
    #                 prompt_end
    #             )
    #     return prompt

    def complete(prompt):
        # query text-davinci-003
        res = openai.Completion.create(
            engine='gpt-3.5-turbo',
            prompt=prompt,
            temperature=0,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return res['choices'][0]['text'].strip()
    if 'llm' not in st.session_state:
        from langchain_openai import OpenAI
        st.session_state['llm'] = OpenAI(api_key=OPENAI_API_KEY ,model_name="gpt-3.5-turbo-instruct")
# llm = OpenAIChat(temperature=0,model_name='gpt-3.5-turbo', api_key= )

        st.session_state['conversation_with_summary'] = ConversationChain(
        llm=st.session_state['llm'], 
        # We set a very low max_token_limit for the purposes of testing.
        memory=ConversationSummaryBufferMemory(llm=st.session_state['llm'], max_token_limit=650)
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown (message["content"])
    if query:=st.chat_input("Enter your query:"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.spinner("Generating response..."):
            with st.chat_message("assistant"):
                query_with_contexts = retrieve(query)
                result=textwrap.fill(str(st.session_state['conversation_with_summary'].predict(input=query_with_contexts)))
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
        # st.write("DONE")