import os
import asyncio
import threading
from loguru import logger

from utils.logging_config import setup_logging
from core.server import Server
from ui.gradio_ui import create_gradio_interface

async def main():
    setup_logging()
    logger.info("TaleWeaver application starting...")

    # Ensure data directory for AIClient exists early
    # AIClient itself also creates this, but good practice for main app.
    default_history_dir = "data/chats"
    os.makedirs(default_history_dir, exist_ok=True)
    logger.info(f"Ensured AI history directory exists at ./{default_history_dir}")

    # Create the server instance
    server_instance = Server()

    # Create the Gradio interface, passing the server instance
    gradio_app = await create_gradio_interface(server_instance)

    gradio_ready = asyncio.Event()

    def run_gradio_app_thread():
        try:
            gradio_app.launch(prevent_thread_lock=True, share=False, inbrowser=True)
            gradio_ready.set() # Signal that Gradio has launched
        except Exception as e:
            logger.critical(f"FATAL: Failed to launch Gradio interface: {e}", exc_info=True)
            # Consider a more graceful shutdown or exit mechanism
            os._exit(1) # Force exit if Gradio fails critically

    # Run Gradio in a separate thread to not block the main asyncio loop
    threading.Thread(target=run_gradio_app_thread, daemon=True).start()

    await gradio_ready.wait() # Wait for Gradio to be ready
    logger.info("Gradio interface is presumed to be up and running.")

    try:
        while True:
            await asyncio.sleep(1) # Keep the main asyncio event loop alive
    except KeyboardInterrupt:
        logger.info("Application shutting down due to KeyboardInterrupt...")
    except asyncio.CancelledError:
        logger.info("Main task cancelled, application shutting down...")
    finally:
        logger.info("Performing cleanup...")
        if hasattr(gradio_app, 'close') and callable(gradio_app.close):
            logger.info("Attempting to close Gradio app...")
            try:
                gradio_app.close() # Try sync close first
                logger.info("Gradio app closed.")
            except Exception as e:
                logger.warning(f"Exception during Gradio close: {e}")
        
        # Close AIClient resources if it has an explicit close method
        if hasattr(server_instance.ai_client, 'close') and callable(server_instance.ai_client.close):
            logger.info("Closing AIClient resources...")
            await server_instance.ai_client.close() # Assuming AIClient has an async close

        logger.info("TaleWeaver application stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user from __main__.")
    except Exception as e:
        logger.critical(f"Unhandled exception in __main__: {e}", exc_info=True)