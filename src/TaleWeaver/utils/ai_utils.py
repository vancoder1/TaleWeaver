import os
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories.file import FileChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage # For explicit message construction
import aiofiles
from loguru import logger

class AIClient:
    def __init__(self, 
                 api_key: str, 
                 model: str, 
                 system_prompt: str,
                 session_id: str = "default_ai_session", # More descriptive default
                 history_dir: str = "data/chats"): # Default history directory
        
        if not api_key:
            logger.warning("AIClient initialized without an API key. API calls will likely fail.")
        if not model:
            logger.warning("AIClient initialized without a model name. Using a default or expecting it to be set.")
            model = "mixtral-8x7b-32768" # A default, though should be configured

        self.model_name = model # Store as model_name for clarity with ChatGroq
        self.api_key = api_key
        self.system_prompt_template = system_prompt # This is the base prompt
        self.current_system_prompt = system_prompt # Actual prompt used, can be updated
        self.session_id = session_id
        self.history_dir = history_dir
        
        os.makedirs(self.history_dir, exist_ok=True) # Ensure base history directory exists
        # Session-specific directory is handled by _get_history_file_path

        self.llm: Optional[ChatGroq] = None
        self.memory: Optional[ConversationSummaryMemory] = None
        self.chain: Optional[RunnableWithMessageHistory] = None # More specific type
        self._initialize_components()

    def _initialize_components(self):
        logger.debug(f"Initializing AIClient components for session: {self.session_id} with model: {self.model_name}")
        try:
            self.llm = ChatGroq(api_key=self.api_key, model_name=self.model_name, streaming=True, temperature=0.7)
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq LLM: {e}", exc_info=True)
            self.llm = None # Ensure llm is None if init fails
            # Optionally, raise the error or handle it to prevent further operations
            return 

        self.memory = self._initialize_memory()
        self.chain = self._initialize_chain()
        logger.info(f"AIClient components initialized for session: {self.session_id}")

    def _initialize_memory(self) -> Optional[ConversationSummaryMemory]:
        if not self.llm:
            logger.error("LLM not initialized, cannot create ConversationSummaryMemory.")
            return None
        try:
            history_file_path = self._get_history_file_path()
            # Ensure the directory for the specific session's history file exists
            os.makedirs(os.path.dirname(history_file_path), exist_ok=True)
            
            # Ensure the history file exists and is valid JSON before FileChatMessageHistory tries to load it
            if not os.path.exists(history_file_path) or os.path.getsize(history_file_path) == 0:
                logger.info(f"History file {history_file_path} not found or empty. Creating with empty list.")
                with open(history_file_path, 'w', encoding='utf-8') as f:
                    f.write('[]') # Valid empty JSON list

            file_chat_history = FileChatMessageHistory(file_path=history_file_path)
            
            return ConversationSummaryMemory(
                llm=self.llm,
                chat_memory=file_chat_history, 
                max_token_limit=450, # Token limit for the generated summary, not for pruning history before summary
                return_messages=True, # Memory will return BaseMessage objects
                # memory_key="history", # Implicitly handled by MessagesPlaceholder("history")
                # input_key="input" # Implicitly handled by input_messages_key="input"
            )
        except Exception as e:
            logger.error(f"Failed to initialize memory for session {self.session_id}: {e}", exc_info=True)
            return None

    def _get_history_file_path(self) -> str:
        session_specific_dir = os.path.join(self.history_dir, self.session_id)
        return os.path.join(session_specific_dir, f"{self.session_id}_chat_history.json")

    def _initialize_chain(self) -> Optional[RunnableWithMessageHistory]:
        if not self.llm or not self.memory:
            logger.error("LLM or Memory not initialized, cannot create chain.")
            return None
        try:
            # Using the current_system_prompt which can be updated dynamically
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.current_system_prompt),
                MessagesPlaceholder(variable_name="history"), # from ConversationSummaryMemory (or any ChatMessageHistory)
                ("human", "{input}"), # 'input' is the key for user's raw message
            ])

            # Core chain: prompt -> LLM -> output parser
            core_chain = prompt_template | self.llm | StrOutputParser()

            # Wrap with message history management
            chain_with_history = RunnableWithMessageHistory(
                runnable=core_chain,
                get_session_history=lambda session_id: self.memory.chat_memory, # type: ignore # self.memory is checked
                input_messages_key="input",    # Key in invoke dict for new human message
                history_messages_key="history", # Key in prompt for historical messages
                # output_messages_key="output" # Not needed if StrOutputParser is last step
            )
            
            # Regarding summarize_messages: ConversationSummaryMemory handles its own summarization.
            # The custom summarize_messages method in the original code was an attempt to manually control this.
            # If ConversationSummaryMemory is used correctly, it should manage summarizing old messages
            # into a summary and keeping recent ones when its buffer (related to max_token_limit) is hit.
            # The RunnablePassthrough.assign for summarize_messages is removed for now to rely on ConversationSummaryMemory.
            # If more explicit control over summarization timing is needed, it's complex.

            return chain_with_history
        except Exception as e:
            logger.error(f"Failed to initialize chain for session {self.session_id}: {e}", exc_info=True)
            return None
    
    async def generate(self, input_text: str) -> str:
        if not self.chain:
            logger.error(f"Chain not available for session {self.session_id}. Cannot generate response.")
            return "AI Error: System not ready. Please try again."
        if not self.memory: # Should not happen if chain is available
            logger.error(f"Memory not available for session {self.session_id}. Cannot generate response.")
            return "AI Error: Memory component not ready."

        try:
            logger.debug(f"Generating AI response for input: '{input_text[:50]}...' in session {self.session_id}")
            # The chain expects a dictionary with the key specified in input_messages_key
            response_content = await self.chain.ainvoke(
                {"input": input_text},
                config={"configurable": {"session_id": self.session_id}} # session_id is passed for history
            )
            # ConversationSummaryMemory's save_context should be called implicitly by the chain.
            # self.memory.save_context({"input": input_text}, {"output": response_content}) # Manual context saving
            logger.debug(f"AI response received for session {self.session_id}: '{str(response_content)[:50]}...'")
            return str(response_content) # Ensure it's a string
            
        except FileNotFoundError: # Should be less likely now with pre-creation of history file
            logger.warning(f"History file possibly deleted mid-session {self.session_id}. Attempting to re-initialize.")
            await self._create_empty_history_file_async() # Ensure file exists
            self._initialize_components() # Re-init might reset state
            if not self.chain: return "AI Error: System had to reset. Please try again."
            response_content = await self.chain.ainvoke(
                {"input": input_text},
                config={"configurable": {"session_id": self.session_id}}
            )
            return str(response_content)
        except Exception as e:
            logger.error(f"Error generating AI response in session {self.session_id}: {e}", exc_info=True)
            return "I'm sorry, an unexpected error occurred with the AI. Please try again or check the logs."

    async def start_new_session(self, session_id: str):
        logger.info(f"Starting new AI session: {session_id}")
        self.session_id = session_id
        # System prompt is not reset here, it's managed by update_system_prompt externally or uses the one from init
        await self._create_empty_history_file_async()
        self._initialize_components() # Re-initializes memory and chain for the new session_id

    async def load_session(self, session_id: str):
        logger.info(f"Loading AI session: {session_id}")
        self.session_id = session_id
        history_file_path = self._get_history_file_path()
        
        # Ensure directory exists before checking file
        os.makedirs(os.path.dirname(history_file_path), exist_ok=True)

        if not os.path.exists(history_file_path) or os.path.getsize(history_file_path) == 0:
            logger.info(f"History file for session {session_id} not found or empty at {history_file_path}. Creating new one.")
            await self._create_empty_history_file_async()
        
        self._initialize_components() # Re-initializes to load messages and use the correct session_id

    async def _create_empty_history_file_async(self):
        file_path = self._get_history_file_path()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write('[]') # Valid empty JSON list for FileChatMessageHistory
            logger.debug(f"Ensured empty history file for session {self.session_id} at {file_path}")
        except Exception as e:
            logger.error(f"Failed to create empty history file {file_path}: {e}", exc_info=True)


    async def get_conversation_history(self) -> List[BaseMessage]: # Returns Langchain message objects
        if self.memory and hasattr(self.memory.chat_memory, 'messages'):
             return self.memory.chat_memory.messages
        logger.warning(f"Could not retrieve conversation history for session {self.session_id}, memory not available.")
        return []

    async def update_system_prompt(self, new_full_prompt: str):
        logger.info(f"Updating system prompt for AI session {self.session_id}.")
        self.current_system_prompt = new_full_prompt # Update the dynamically used prompt
        # Re-initialize the chain to use the new system prompt
        # This is important because the prompt is part of the chain definition.
        self.chain = self._initialize_chain()
        if self.chain:
            logger.debug(f"Chain re-initialized with new system prompt for session {self.session_id}.")
        else:
            logger.error(f"Failed to re-initialize chain with new system prompt for session {self.session_id}.")

    async def close(self):
        """Explicitly close any open resources, like network clients."""
        logger.info(f"Closing AIClient resources for session {self.session_id}...")
        if self.llm and hasattr(self.llm, 'client') and hasattr(self.llm.client, 'aclose'):
            try:
                # Assuming httpx.AsyncClient or similar that has an async close
                await self.llm.client.aclose()
                logger.debug("LLM's internal async client closed.")
            except Exception as e:
                logger.warning(f"Error closing LLM's async client: {e}", exc_info=True)
        elif self.llm and hasattr(self.llm, 'client') and hasattr(self.llm.client, 'close'):
             try:
                # Fallback for synchronous client
                self.llm.client.close()
                logger.debug("LLM's internal sync client closed.")
             except Exception as e:
                logger.warning(f"Error closing LLM's sync client: {e}", exc_info=True)

    def __del__(self):
        pass