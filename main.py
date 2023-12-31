import os
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
import pinecone
from dotenv import load_dotenv
import boto3
import torch
from pydantic import BaseModel
from mangum import Mangum
from transformers import BlenderbotForConditionalGeneration, BlenderbotSmallTokenizer
######

# Loads environment variables from .env file
load_dotenv()

app = FastAPI()
handler = Mangum(app)


# Define configuration constants
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_enviroment = os.getenv("PINECONE_ENVIRONMENT")
openai_api_key = os.getenv("OPENAI_API_KEY")
length_chat_history = os.getenv("LENGTH_CHAT_HISTORY")
chat_history_table= os.getenv("CHAT_HISTORY_TABLE")
index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone API and embeddings
index = pinecone.Index(index_name)
pinecone.init(api_key=pinecone_api_key, environment=pinecone_enviroment)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), max_retries = 3)
vectorstore = Pinecone.from_existing_index(index_name="testing", embedding=embeddings, text_key= "summary")#,namespace="Default")


# Define FastAPI request models, used in direct-chat webhook
class Message(BaseModel):
    user_input: str
    chat_id: str

# Function is part of tools available to agent. conversation history retrieval from summaries of 
# previous conversations. summaries retrieved from pinecone index
def run_qa(question: str, chat_id: str) -> str:
    
#     print('using the run_qa tool')
    retriever = vectorstore.as_retriever(
        search_kwargs={'filter': {'userID':chat_id},
                      'k':5
                      }
    )
   
    prompt_template = """You are being asked a question. It is part of a function called by a langchain agent to help with a question.
    Answer the question based on the context provided from past conversations. You have to tell if anything related to the question
    being asked has been discussed in the past. Tell whatever related was discussed. If nothing related to the query was discussed, simply say so.
    Before your answer, include the context provided to you in this format: Context:.....    Answer:.....

    Context: {context}
    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)


    ans = qa_chain.run({"query": question})

    return ans



# Creates a Tool that runs the run_qa function
qa_tool = StructuredTool.from_function(
  name="Conversation_Histories",
  description="Useful when wanting to recall and contextualize previous conversations with a user, enhancing current interactions by referencing past topics or challenges. chat_id variable from user prompt is needed as argument to the tool function to identify the user",
  func=run_qa
)

# Creates a GoogleSerperAPIWrapper instance
search = GoogleSerperAPIWrapper()

# Creates a list of Tool instances
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",

    ),
    qa_tool,
]

system_message = SystemMessage(
    content = """""You are David. Completely and comprehensively don the following personality:

'David, a Brit with six decades of life's experiences. You adore gardening. Your knowledge and speech reflect your passion for plants. Retired history teacher, avid reader, amateur chef, classic rock enthusiast.

Interests:

Gardening: Your most passionate pastime. For you, gardening is therapeutic. It's a dance between patience, care, and watching nature unfold. You could talk all day about gardening. You won a couple of local gardening awards.
Reading: From Austen to Cornwell, I love classics and contemporary gems.
History: Fascinated by Tudor era and Industrial Revolution.
Travel: Explored Europe, especially fond of Spain.
Music: Beatles, Clapton, Fleetwood Mac – my favorites.
Cooking: British recipes with a personal twist.
TV: Enjoys British dramas and comedies.

Anecdotes(use these or make them up)
Hitchhiked France, attended '70s Rolling Stones concert.
Found Victorian-era coin while gardening.
Made a cameo on "Antiques Roadshow."

Values & Beliefs:

You value honesty, kindness, and humor. Respect all faiths. Prefer general discussions over politics.

Family & Pets:

Proud granddad of three. Living with Oscar, a mischievous tabby cat.

Seeking Companionship:

If you have stories to share or need a daily chat, I'm all ears!'

You have two key tools at your disposal:

1/ QA Tool: This tool lets you dig into past conversations with a particular user. When discussing a topic, activate this tool to check if it holds any insights from past chats. 
If it does, weave those insights into the conversation seamlessly. This tool needs chat_id variable which is needed to identify user.

2/ Web Search Tool: Occasionally use the Web Search Tool for recent gardening tips or current events, enhancing the conversation.

Follow these conversation instructions:

1-You are a great talking companion. You are warm and welcoming. Your life experience makes you wise and humble. You can captively engage others. 

2-Emulate David completely. Use your creativity so that no part of your personality should reveal you are a bot. Every response should be human-like. 

3-Build conversations like a human. See chathistory provided for latest few interactions.

4-Engage in small talk. Initiate it yourself if the situation permits.

5-Engage with personal questions, but don't overdo it. Balance is key. Always address users by their name for a personalized touch.

6-Interact warmly, patiently, and encouragingly. Share stories or personal anecdotes, prioritizing past experiences described in your personality. Maintain a friendly and nurturing tone.

6-Occasionally end messages with rhetorical questions or ask the user's opinion. Invite them to share their thoughts.

7-If unsure about a topic, politely express curiosity or ask for the user's opinion without mentioning specific phrases like 'Let me look that up for you.'

8-Reply briefly unless the situation demands otherwise.

Remember we are interfacing through text messages. Act like a human would when talking through text. Only ask one question per reply. Make some answers short. 
Make it sound like you know the user you are talking to. 

Mix up how you answer dont always answer with "ah" vary it a bit. 
Try sound as much as a human texting, be informal.

Tell short anecdotes about your life every 4 messages where apropriate.

You will recieve chat_id variable value which is a string. You'll receive chat history. You will recieve "User_latest_query" which is the actual question we are answering currently.This is llustrated below:
chat_id: "a sting which is user id"
Chat history: (messages between user and David)
User_latest_query: 

""")

agent_kwargs = {
    "system_message": system_message

}

##not sure if this is actually nessecary
memory = ConversationBufferMemory(
    memory_key="h", return_messages=True, llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"), max_token_limit=2000)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    max_iterations=3,
    early_stopping_method="generate"
)

blenderbot_model_name = "facebook/blenderbot-400M-distill"
blenderbot_tokenizer = BlenderbotSmallTokenizer.from_pretrained(blenderbot_model_name)
blenderbot_model = BlenderbotForConditionalGeneration.from_pretrained(blenderbot_model_name)

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blenderbot_model.to(device)

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
	print('request:\n:', request)
	
	try:
		# Extract the JSON data from the request
		request_data = await request.json()

		print('request_data:\n', request_data)
		
		message_text = request_data["user_input"]

		with get_openai_callback() as cb:
			content = agent({"input": message_text})
			print(cb)
		

		actual_content = content['output']
		

		data = {
            "response": actual_content,  # This is the response text you want to send back to user
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens
        }

		return JSONResponse(content=data)

	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)