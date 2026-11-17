import os 
from dotenv import load_dotenv
import requests
from langchain.tools import tool 

load_dotenv()
BASE_API_URL = os.environ['BASE_API_URL']

order_status_tool_description = """Returns the order details and status for the given order ID.
Returns a dictionary with the products ordered, the price and quantity of each, order date, 
and order status, for example, processing, shipped, delivered, returned. 
If the order has been delivered or returned, the delivery date will be included. 
If there is an error getting the order details and status, 
a dictionary with one key 'error' and an error message will be returned. """

@tool(description=order_status_tool_description)
def get_order_status(order_id: int) -> dict:
    url = f'{BASE_API_URL}/orders/status/{order_id}'
    print(url)
    resp = requests.get(url)
    if resp.status_code != 200:
        return {'error': 'Error calling order status API'}
    else:
        return resp.json()