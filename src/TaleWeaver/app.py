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
    try:
        os.makedirs("data", exist_ok=True) 
        logger.info("Ensured base 'data' directory exists.")
    except Exception as e:
        logger.warning(f"Could not create 'data' directory: {e}")

    # Create the server instance
    server_instance = Server() 
    logger.info("Server instance created.")

    # Create the Gradio interface, passing the server instance
    gradio_app = await create_gradio_interface(server_instance)
    logger.info("Gradio interface definition created.")

    gradio_ready = asyncio.Event()

    def run_gradio_app_thread():
        nonlocal gradio_app # Ensure it's the same app instance
        try:
            # Launch Gradio. prevent_thread_lock=True is good for dev, but can be removed
            # if causing issues with asyncio event loop management in some environments.
            # share=False is default and safer. inbrowser=True is convenient.
            gradio_app.launch(share=False, inbrowser=True) 
            gradio_ready.set() # Signal that Gradio has launched (or at least the launch call returned)
        except Exception as e:
            logger.critical(f"FATAL: Failed to launch Gradio interface: {e}", exc_info=True)
            os._exit(1) # Force exit if Gradio fails critically

    # Run Gradio in a separate thread to not block the main asyncio loop
    gradio_thread = threading.Thread(target=run_gradio_app_thread, daemon=True)
    gradio_thread.start()
    logger.info("Gradio launch thread started.")

    await gradio_ready.wait() # Wait for Gradio to signal it's (likely) up
    logger.info("Gradio interface is presumed to be up and running.")

    try:
        while True:
            # Keep the main asyncio event loop alive to handle server-side async tasks
            # if any were to run independently of Gradio callbacks.
            await asyncio.sleep(3600) # Sleep for a long time, effectively waiting for interrupt
    except KeyboardInterrupt:
        logger.info("Application shutting down due to KeyboardInterrupt...")
    except asyncio.CancelledError:
        logger.info("Main task cancelled, application shutting down...")
    finally:
        logger.info("Performing cleanup...")
        
        # Close Gradio app first
        if gradio_app is not None and hasattr(gradio_app, 'close') and callable(gradio_app.close):
            logger.info("Attempting to close Gradio app...")
            try:
                # Gradio's close method might be synchronous.
                # Running it in executor if it blocks, but usually it's quick.
                gradio_app.close() 
                logger.info("Gradio app closed.")
            except Exception as e:
                logger.warning(f"Exception during Gradio close: {e}", exc_info=True)
        
        # Close Server resources (which includes AIClient resources)
        if server_instance is not None and hasattr(server_instance, 'close_server_resources') and callable(server_instance.close_server_resources):
            logger.info("Closing server resources (including AIClient)...")
            await server_instance.close_server_resources()
        
        logger.info("TaleWeaver application stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This might not be reached if the KeyboardInterrupt is caught inside main's try block
        logger.info("Application terminated by user from __main__ (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Unhandled exception in __main__ execution: {e}", exc_info=True)