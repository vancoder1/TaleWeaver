import os
import asyncio
import threading
from loguru import logger
from utils.logging_config import setup_logging
from core.server import Server
from ui.gradio_ui import create_gradio_interface

async def main():
    setup_logging()
    logger.info("TaleWeaver starting...")
    try:
        os.makedirs("data/chats", exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create data/chats directory: {e}")

    server_instance = None
    try:
        server_instance = Server()
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize Server: {e}", exc_info=True); os._exit(1)

    gradio_app = await create_gradio_interface(server_instance)
    gradio_ready = asyncio.Event()

    def run_gradio_app_thread():
        try:
            gradio_app.launch(share=False, inbrowser=True)
            gradio_ready.set()
        except Exception as e:
            logger.critical(f"FATAL: Failed to launch Gradio: {e}", exc_info=True)
            if server_instance: asyncio.run(server_instance.close_server_resources())
            os._exit(1)

    threading.Thread(target=run_gradio_app_thread, daemon=True).start()

    try: await asyncio.wait_for(gradio_ready.wait(), timeout=30.0)
    except asyncio.TimeoutError: logger.error("Gradio readiness timeout.")

    try:
        while True: await asyncio.sleep(3600) # Keep main loop alive
    except (KeyboardInterrupt, asyncio.CancelledError): logger.info("Shutting down...")
    finally:
        logger.info("Cleaning up...")
        try: gradio_app.close()
        except Exception as e: logger.warning(f"Gradio close error: {e}")
        if server_instance:
            try: await server_instance.close_server_resources()
            except Exception as e: logger.error(f"Server close error: {e}")
        logger.info("TaleWeaver stopped.")

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: logger.info("Terminated by user.")
    except Exception as e: logger.critical(f"Unhandled exception: {e}", exc_info=True); os._exit(1)