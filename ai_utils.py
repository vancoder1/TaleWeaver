import os
import json
from typing import AsyncGenerator, Optional, List
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories.file import FileChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import aiofiles
import logging

class AIClient:
    def __init__(self, 
                 api_key: str, 
                 model: str, 
                 system_prompt: str = "You are guiding an immersive adventure",
                 session_id: str = "taleweaver_session",
                 history_dir: str = "data/chats"):
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.session_id = session_id
        self.history_dir = history_dir
        os.makedirs(self.history_dir, exist_ok=True)
        self._initialize_components()
        self.logger = logging.getLogger(__name__)

    def _initialize_components(self):
        self.llm = ChatGroq(api_key=self.api_key, model=self.model, streaming=True)
        self.memory = self._initialize_memory()
        self.chain = self._initialize_chain()

    def _initialize_memory(self):
        file_history = FileChatMessageHistory(self._get_history_file_path())
        return ConversationSummaryBufferMemory(
            llm=self.llm,
            chat_memory=file_history,
            max_token_limit=8192,
            return_messages=True,
        )

    def _get_history_file_path(self) -> str:
        return os.path.join(self.history_dir, f"{self.session_id}.json")

    def _initialize_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        return RunnableWithMessageHistory(
            chain,
            lambda session_id: self.memory.chat_memory,
            input_messages_key="input",
            history_messages_key="history"
        )  
    
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
        # Ensure proper cleanup of resources
        if hasattr(self, 'llm'):
            self.llm.client.close()