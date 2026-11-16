""" Text chat interface for agent """

from agent import generate_response 
import rich 
from rich.markdown import Markdown
from uuid import uuid4


def agent_chat(debug=True):

    print('Welcome to the Plant Store AI assistant! How can I help?')

    # create a unique, hard (impossible?) to guess conversation ID for each conversation
    thread_id = uuid4()

    while True:
        user_message = input('>> ')
        if not user_message:
            print('Bye!')
            break
        ai_message = generate_response(user_message, thread_id)
        rich.print(Markdown(ai_message))

if __name__ == '__main__':
    agent_chat(debug=True)

