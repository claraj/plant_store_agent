import os 
from dotenv import load_dotenv
import requests
from langchain.tools import tool 

load_dotenv()
BASE_API_URL = os.environ['BASE_API_URL']


search_plants_by_price_tool_description = """Provide a min_price and/or a max_price. 
Returns a list of all matching plants and their prices, description, scientific name, 
care needs, within the given price range.
"""

@tool(description=search_plants_by_price_tool_description)
def search_plants_by_price(query: dict) -> list:
    print('CALLING SEARCH PLANTS BY PRICE', query)

    url = f'{BASE_API_URL}/search/price'
    
    resp = requests.get(url, query)
    if resp.status_code != 200:
        print(resp)
        return 'Error calling plant price search API'
    else:
        return resp.json()
    

search_plants_by_name_tool_description = """Provide a plant name. 
Returns a list of all matching plants and their prices, 
description, scientific name, care needs, whose common name contains the name entered. 
Makes partial matches e.g. "Bird" will match "Bird of Paradise"""

@tool(description=search_plants_by_name_tool_description)
def search_plants_by_name(name: str) -> list:
    print('CALLING SEARCH PLANTS BY NAME', name)

    url = f'{BASE_API_URL}/search/name'

    query = {'name': name}
    response = requests.get(url, query)
    if response.status_code != 200:
        return 'Error calling plant name search'
    else:
        return response.json()



search_plants_by_care_level_tool_description = """Provide a care level or levels from the list 'easy', 'medium' or difficult.
Can search for multiple care levels by joining with ; for example 'easy;medium'. Returns a list of all matching plants 
and their prices, description, scientific name, and care needs."""

@tool(description=search_plants_by_care_level_tool_description)
def search_plants_by_care_level(care_level: str) -> list:
    print('CALLING SEARCH PLANTS BY CARE LEVEL', care_level)

    url = f'{BASE_API_URL}/search/care'

    query = {'care_level': care_level}
    response = requests.get(url, query)
    if response.status_code != 200:
        return 'Error calling plant care level search'
    else:
        return response.json()