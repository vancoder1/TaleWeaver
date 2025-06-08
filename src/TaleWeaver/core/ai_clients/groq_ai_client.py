import os
import asyncio
from loguru import logger
from typing import List, Optional, Any

from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories.file import FileChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage

from .ai_provider_interface import AIProviderInterface

class GroqAIClient(AIProviderInterface):
    def __init__(self, model: str, system_prompt: str, session_id: str, history_dir: str,
                 api_key: Optional[str] = None):
        super().__init__(model=model, system_prompt=system_prompt, session_id=session_id,
                         history_dir=history_dir, api_key=api_key)
        if not self.api_key: raise ValueError("Groq API key is required.")
        self._initialize_components()

    def _initialize_components(self) -> None:
        try:
            self.llm = ChatGroq(api_key=self.api_key, model_name=self.model_name)
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq LLM: {e}", exc_info=True); raise
        self.memory = self._initialize_memory()
        self.chain = self._initialize_chain()
        logger.info(f"GroqAIClient components initialized for session: {self.session_id}")

    def _initialize_memory(self) -> Optional[ConversationSummaryMemory]:
        if not self.llm: return None
        try:
            history_file_path = self._get_history_file_path()
            os.makedirs(os.path.dirname(history_file_path), exist_ok=True)
            if not os.path.exists(history_file_path) or os.path.getsize(history_file_path) == 0:
                with open(history_file_path, 'w', encoding='utf-8') as f: f.write('[]')
            
            file_chat_history = FileChatMessageHistory(file_path=history_file_path)
            return ConversationSummaryMemory(
                llm=self.llm, chat_memory=file_chat_history,
                max_token_limit=450, # Hardcoded as additional_params is removed
                return_messages=True,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Groq memory for {self.session_id}: {e}", exc_info=True)
            return None

    def _initialize_chain(self) -> Optional[RunnableWithMessageHistory]:
        if not self.llm or not self.memory: return None
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.current_system_prompt),
                MessagesPlaceholder(variable_name="history"), ("human", "{input}"),
            ])
            core_chain = prompt | self.llm | StrOutputParser()
            return RunnableWithMessageHistory(
                runnable=core_chain,
                get_session_history=lambda _: self.memory.chat_memory if self.memory else None,
                input_messages_key="input", history_messages_key="history",
            )
        except Exception as e:
            logger.error(f"Failed to initialize Groq chain for {self.session_id}: {e}", exc_info=True)
            return None

    async def generate(self, input_text: str) -> str:
        if not self.chain or not self.memory: return "AI Error: Groq system not ready."
        try:
            response = await self.chain.ainvoke(
                {"input": input_text}, config={"configurable": {"session_id": self.session_id}}
            )
            return str(response)
        except FileNotFoundError: # History file deleted mid-session
            logger.warning(f"History file missing for {self.session_id}. Re-initializing.")
            await self._create_empty_history_file_async()
            self._initialize_components() # Re-init all
            if not self.chain: return "AI Error: Groq system reset. Try again."
            response = await self.chain.ainvoke(
                {"input": input_text}, config={"configurable": {"session_id": self.session_id}}
            )
            return str(response)
        except Exception as e:
            logger.error(f"Error generating Groq response for {self.session_id}: {e}", exc_info=True)
            return "AI Error (Groq): Unexpected error."

    async def update_system_prompt(self, new_full_prompt: str) -> None:
        self.current_system_prompt = new_full_prompt
        self.chain = self._initialize_chain() # Re-initialize chain with new prompt
        if not self.chain: logger.error(f"Failed to re-initialize Groq chain for {self.session_id}.")

    async def start_new_session(self, session_id: str) -> None:
        self.session_id = session_id
        await self._create_empty_history_file_async()
        self._initialize_components()

    async def load_session(self, session_id: str) -> None:
        self.session_id = session_id
        history_file_path = self._get_history_file_path()
        os.makedirs(os.path.dirname(history_file_path), exist_ok=True)
        if not os.path.exists(history_file_path) or os.path.getsize(history_file_path) == 0:
            await self._create_empty_history_file_async()
        self._initialize_components()

    async def get_conversation_history(self) -> List[BaseMessage]:
        if self.memory and hasattr(self.memory.chat_memory, 'messages'):
             return self.memory.chat_memory.messages
        return []

    async def close(self) -> None:
        logger.info(f"Closing GroqAIClient for session {self.session_id}...")
        if self.llm and hasattr(self.llm, 'client'):
            client_to_close = self.llm.client
            close_method = getattr(client_to_close, 'aclose', getattr(client_to_close, 'close', None))
            if close_method:
                try:
                    if asyncio.iscoroutinefunction(close_method): await close_method()
                    else: close_method()
                except Exception as e: logger.warning(f"Error closing Groq client: {e}")