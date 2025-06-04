import gradio as gr
from utils import constants
from core.server import Server
from loguru import logger 

# Helper to add system messages to chat history UI
def _add_system_message_to_history(current_history, message_text, source_tag="System"):
    system_message = (source_tag, message_text)
    # Add error to history only if it's not the last message already
    if not current_history or current_history[-1] != system_message:
        current_history.append(system_message)
    return current_history

async def create_gradio_interface(server: Server):
    with gr.Blocks(theme=gr.themes.Soft(), title="TaleWeaver") as interface:
        gr.Markdown("# TaleWeaver")

        def get_char_name(): return server.character.name
        def get_char_backstory(): return server.character.backstory

        with gr.Tab("Setup"):
            with gr.Row():
                with gr.Column(scale=2):
                    session_name_input = gr.Textbox(label="New Session Name", placeholder="e.g., MyEpicAdventure")
                    setting_input = gr.Textbox(label="Game Setting / World Description", lines=3, placeholder="e.g., A medieval fantasy kingdom...")
                    language_dropdown = gr.Dropdown(label="Select Language", choices=list(constants.LANGUAGES.keys()), value="english")
                    start_button = gr.Button("Start New Game", variant="primary")
            
            with gr.Row():
                 with gr.Column():
                    current_sessions = server.get_available_sessions()
                    load_dropdown = gr.Dropdown(label="Load Previous Session", choices=current_sessions, value=current_sessions[0] if current_sessions else None, interactive=True)
                    load_button = gr.Button("Load Selected Session")
            setup_status = gr.Textbox(label="Setup Status", interactive=False, lines=1) # Reduced lines

        with gr.Tab("Game"):
            chat_history_output = gr.Chatbot(label="Game History", height=600, bubble_full_width=False)
            with gr.Row():
                mode_dropdown = gr.Dropdown(label="Action Mode", choices=constants.MODES, value="Say", scale=1)
                action_input = gr.Textbox(label="Your Action", placeholder="Type what you say or do...", scale=5, show_label=False)
                submit_action_button = gr.Button("Submit Action", variant="primary", scale=1)
            story_button = gr.Button("Continue Story (AI Only)", scale=1)
            game_status_output = gr.Textbox(label="Game Status", value="No active game session.", interactive=False, lines=1) # Reduced lines

        with gr.Tab("Settings"):
            with gr.Tabs():
                with gr.TabItem("Character Configuration"):
                    gr.Markdown("### Define Your Character")
                    character_name_input = gr.Textbox(label="Character Name", value=get_char_name)
                    character_backstory_input = gr.Textbox(label="Character Backstory", lines=5, value=get_char_backstory)
                    update_character_button = gr.Button("Save Character Details", variant="primary")
                    character_config_status_output = gr.Textbox(label="Status", interactive=False, lines=1)

                with gr.TabItem("General AI Configuration"):
                    gr.Markdown("### AI Model and Prompts")
                    api_key_input = gr.Textbox(label="GROQ API Key", value=lambda: server.json_handler.get_setting('groq_api_key', ''), type="password")
                    model_input = gr.Textbox(label="AI Model Name", value=lambda: server.json_handler.get_setting('ai_settings.model', ''))
                    system_prompt_input = gr.Textbox(label="Base System Prompt (character & setting added automatically)", value=lambda: server.json_handler.get_setting('ai_settings.system_prompt', ''), lines=8)
                    update_config_button = gr.Button("Update General AI Configuration", variant="stop")
                    config_status_output = gr.Textbox(label="Status", interactive=False, lines=1)

        # --- Setup Tab Logic ---
        async def start_game_gradio(session, setting, lang_key):
            if not session.strip():
                return "Session name required.", server.message_history, "Session name cannot be empty.", gr.update()
            
            lang_code = constants.LANGUAGES.get(lang_key, "en")
            start_status = await server.start_game(session, setting, lang_code)
            updated_sessions = server.get_available_sessions()
            game_status_msg = f"Game '{session}' started for {server.character.name}. Lang: {lang_key}."
            return start_status, [], game_status_msg, gr.update(choices=updated_sessions, value=session)

        start_button.click(
            start_game_gradio,
            inputs=[session_name_input, setting_input, language_dropdown],
            outputs=[setup_status, chat_history_output, game_status_output, load_dropdown]
        ).then(lambda: (gr.update(value=""), gr.update(value="")), outputs=[session_name_input, setting_input])

        async def load_session_gradio(session_to_load, lang_key):
            if not session_to_load:
                 return "No session selected.", server.message_history, "Select a session to load.", gr.update()
            
            load_status = await server.load_session(session_to_load, lang_key)
            game_status_msg = f"Loaded '{session_to_load}' for {server.character.name}. Lang: {lang_key}."
            return load_status, server.message_history, game_status_msg, gr.update(value=session_to_load)

        load_button.click(
            load_session_gradio,
            inputs=[load_dropdown, language_dropdown],
            outputs=[setup_status, chat_history_output, game_status_output, load_dropdown]
        )

        # --- Game Tab Logic ---
        async def handle_chat_action(action_text, mode_choice, current_chat_history):
            if not server.current_session:
                msg = server.json_handler.get_setting("user_messages.error_not_in_game_session", "System: No active game.")
                return _add_system_message_to_history(current_chat_history, msg), "", "Not in a game session."
            if not server.character.name: # Should always be true after server init
                msg = server.json_handler.get_setting("user_messages.error_missing_character_setup", "System: Character not set up.")
                return _add_system_message_to_history(current_chat_history, msg), "", "Character not set up."

            new_messages = await server.action(action_text, mode_choice)
            updated_history = current_chat_history + new_messages
            return updated_history, gr.update(value=""), f"Action by {server.character.name} processed."

        submit_action_button.click(
            handle_chat_action,
            inputs=[action_input, mode_dropdown, chat_history_output],
            outputs=[chat_history_output, action_input, game_status_output]
        )
        action_input.submit(
            handle_chat_action,
            inputs=[action_input, mode_dropdown, chat_history_output],
            outputs=[chat_history_output, action_input, game_status_output]
        )

        async def continue_story_gradio(current_chat_history):
            if not server.current_session:
                msg = server.json_handler.get_setting("user_messages.error_not_in_game_session", "No active game.")
                return _add_system_message_to_history(current_chat_history, msg), "Not in a game session."
            
            new_messages = await server.action("", "Story")
            return current_chat_history + new_messages, "Story continued by AI."

        story_button.click(
            continue_story_gradio,
            inputs=[chat_history_output],
            outputs=[chat_history_output, game_status_output]
        )

        # --- Settings Tab Logic ---
        async def update_character_config_gradio(char_name, char_backstory):
            _, status_msg = await server.update_character_config(char_name, char_backstory)
            # Refresh display values in case of normalization/trimming by server (though not currently done)
            return status_msg, gr.update(value=server.character.name), gr.update(value=server.character.backstory)

        update_character_button.click(
            update_character_config_gradio,
            inputs=[character_name_input, character_backstory_input],
            outputs=[character_config_status_output, character_name_input, character_backstory_input]
        )

        async def update_general_config_gradio(api_key, model, base_prompt):
            settings = {
                "groq_api_key": api_key, "ai_settings.model": model, "ai_settings.system_prompt": base_prompt
            }
            status_msg = await server.update_config(settings)
            return ( status_msg,
                gr.update(value=server.json_handler.get_setting('groq_api_key')),
                gr.update(value=server.json_handler.get_setting('ai_settings.model')),
                gr.update(value=server.json_handler.get_setting('ai_settings.system_prompt'))
            )

        update_config_button.click(
            update_general_config_gradio,
            inputs=[api_key_input, model_input, system_prompt_input],
            outputs=[config_status_output, api_key_input, model_input, system_prompt_input]
        )

        def refresh_ui_on_load():
            sessions = server.get_available_sessions()
            return ( gr.update(choices=sessions, value=sessions[0] if sessions else None),
                     gr.update(value=get_char_name()), gr.update(value=get_char_backstory()) )

        interface.load(refresh_ui_on_load, outputs=[load_dropdown, character_name_input, character_backstory_input])
    return interface