import os 
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.messages import HumanMessage 
from langgraph.checkpoint.memory import InMemorySaver

from tools.order_status_tools import get_order_status
from tools.plant_search_tools import search_plants_by_care_level, search_plants_by_name, search_plants_by_price
from return_policy_rag.return_policy_rag import return_policy_query

from plant_search_rag.plant_search_rag import plant_query

load_dotenv()
LLM_MODEL = os.environ['AGENT_LLM_MODEL']

agent_system_prompt = """You are a friendly, helpful assistant for our houseplant store. 
Only provide information that exists in the store's database. 
If the user asks about plants we do not sell or anything not plant-related, politely refuse to answer, 
and remind them that you can only help with houseplants from our plant store. 
Keep all the responses simple and non-technical, and focus on practical and easy to understand guidance."""

agent = create_agent(
    model=LLM_MODEL,
    tools=[
        get_order_status, 
        search_plants_by_price,
        search_plants_by_name,
        search_plants_by_care_level,
        return_policy_query,
        plant_query
        ],
    system_prompt=agent_system_prompt,
    checkpointer=InMemorySaver())


def generate_response(user_message, thread_id):

    user_message = user_message.strip()
    human_message = HumanMessage(user_message)

    config = {
        'configurable': {
            'thread_id': thread_id
            }
        }

    response = agent.invoke(
        { 'messages': [human_message] }, 
        config=config)
    
    messages = response['messages']
    message_list = [m.pretty_repr() for m in messages]
    ai_message = messages[-1]   # The last message in the thread
    ai_message_text = ai_message.text
    return ai_message_text