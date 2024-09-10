import os
import logging
from typing import List
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories.file import FileChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import aiofiles

class AIClient:
    def __init__(self, 
                 api_key: str, 
                 model: str, 
                 system_prompt: str,
                 session_id: str = "unidentified",
                 history_dir: str = "data/chats"):
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.session_id = session_id
        self.history_dir = history_dir
        os.makedirs(self.history_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._initialize_components()

    def _initialize_components(self):
        self.llm = ChatGroq(api_key=self.api_key, model=self.model, streaming=True)
        self.memory = self._initialize_memory()
        self.chain = self._initialize_chain()

    def _initialize_memory(self):
        file_history = FileChatMessageHistory(self._get_history_file_path())
        return ConversationSummaryMemory(
            llm=self.llm,
            chat_memory=file_history,
            max_token_limit=450,
            return_messages=True,
        )

    def _get_history_file_path(self) -> str:
        session_dir = os.path.join(self.history_dir, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        return os.path.join(session_dir, f"{self.session_id}.json")

    def _get_session_dir(self) -> str:
        return os.path.join(self.history_dir, self.session_id)

    def _initialize_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        chain = prompt | self.llm | StrOutputParser()
        chain_with_message_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.memory.chat_memory,
            input_messages_key="input",
            history_messages_key="history"
        )
        return (RunnablePassthrough.assign(messages_summarized=self.summarize_messages)
            | chain_with_message_history
        )
    
    def summarize_messages(self, chain_input):
        stored_messages = self.memory.chat_memory.messages
        if len(stored_messages) == 0:
            return False
        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                (
                    "user",
                    "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
                ),
            ]
        )
        summarization_chain = summarization_prompt | self.llm
        summary_message = summarization_chain.invoke({"history": stored_messages})
        self.memory.chat_memory.clear()
        self.memory.chat_memory.add_message(summary_message)
        return True
    
    async def generate(self, input_text: str) -> str:
        try:
            return await self.chain.ainvoke(
                {"input": input_text},
                config={"configurable": {"session_id": self.session_id}}
            )
        except FileNotFoundError:
            self.logger.warning(f"History file not found for session {self.session_id}. Creating a new one.")
            await self._create_empty_history_file()
            return await self.chain.ainvoke(
                {"input": input_text},
                config={"configurable": {"session_id": self.session_id}}
            )
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return "I'm sorry, but I encountered an error while processing your request. Please try again."

    async def start_new_session(self, session_id: str):
        self.session_id = session_id
        await self._create_empty_history_file()
        self._initialize_components()

    async def load_session(self, session_id: str):
        self.session_id = session_id
        if not os.path.exists(self._get_history_file_path()):
            self.logger.info(f"Creating new history file for session {session_id}")
            await self._create_empty_history_file()
        self._initialize_components()

    async def _create_empty_history_file(self):
        file_path = self._get_history_file_path()
        async with aiofiles.open(file_path, 'w') as f:
            await f.write('[]')

    async def get_conversation_history(self) -> List[dict]:
        return [message.dict() for message in self.memory.chat_memory.messages]

    async def update_system_prompt(self, new_prompt: str):
        self.system_prompt = new_prompt
        self._initialize_components()

    def __del__(self):
        if hasattr(self, 'llm'):
            self.llm.client.close()