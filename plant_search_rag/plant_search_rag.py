import json 
import os 
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool 

load_dotenv()

EMBEDDING_MODEL = os.environ['EMBEDDING_MODEL']
RAG_LLM_MODEL = os.environ['RAG_LLM_MODEL']

plant_database_file = 'plant_search_rag/plants.json'

plant_rag_system_prompt = """Answer the customer's question using only the information provided in the context.
If the context does not contain enough information to answer the question reply with 'I don't know.' 
Do not add outside information.
If the customer's question is not related to the store's houseplants or houseplant-related topics. 
Explain that you can only answer questions regarding the plant store."""

plant_rag_tool_docstring = """Answer general questions about plant care and plant attributes for plants we sell.
And, find plants that match customer requirements, for example, care levels, size, amount of sun or light, 
water requirements, temperature, humidity, scientific name, or care tips.
if the answer cannot be provided then "I don't know" will be returned."""


def create_retriever(generate_embeddings=False):

    collection_name = 'plant_details'
    persist_directory = './plant_embeddings'

    def plant_json_to_text(plant):

        return f"""
        Plant name: {plant['name']} 
        Description: {plant['description']} 
        Scientific name: {plant['scientific_name']}
        Care, light levels: {plant['care']['light']}
        Care, water needs: {plant['care']['water']}
        Care, soil: {plant['care']['soil']}
        Care, prefered temperature and humidity: {plant['care']['temperature_and_humidity']}
        Care, tips: {plant['care']['tips']}
        """

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    if generate_embeddings:
        data = json.load(open(plant_database_file))
        texts = [plant_json_to_text(plant) for plant in data]
        documents = [Document(page_content=text) for text in texts]

        vector_store = Chroma.from_documents(
            collection_name=collection_name,
            documents=documents,    # ids are generated 
            embedding=gemini_embeddings,
            persist_directory=persist_directory
            )
        
    else:  # load existing database 
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=gemini_embeddings,
            persist_directory=persist_directory
        )


    # query = 'What is a good plant for a sunny window?'
    # print('MMR results:')
    # results = vector_store.max_marginal_relevance_search(query, k=2)
    # for res in results:
    #     print(f"{res.page_content}\n")

    # print('Similarity results:')
    # results = vector_store.similarity_search(query, k=2)
    # for res in results:
    #     print(f"{res.page_content}\n")

    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 15}) 
    return retriever


def create_rag_chain(retriever):

    rag_prompt = PromptTemplate(
        template="""{instructions}
        
        Use this context to answer the question:
        {context}

        Customer question:
        {question}

        Response:
        """,
        input_variables=['instructions', 'question', 'context']
    )

    llm = ChatGoogleGenerativeAI(model=RAG_LLM_MODEL, temperature=0.5)

    plant_rag_chain = (
        {
            'instructions': lambda x: plant_rag_system_prompt, 
            'context': retriever, 
            'question': RunnablePassthrough()
            }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return plant_rag_chain



@tool(description=plant_rag_tool_docstring)
def plant_query(query: str) -> str:
    return plant_rag_chain.invoke(query)


retriever = create_retriever(True)
plant_rag_chain = create_rag_chain(retriever)


if __name__ == '__main__':
    query = 'what is an easy plant for a sunny window'
    # query = 'how big is an elephant?'
    # query = 'are hoyas hard to look after'
    # query = 'does a hoya plant like humidity'
    print('Query:', query)
    rag_result = plant_rag.invoke(query)

    print('Rag result:', rag_result)

