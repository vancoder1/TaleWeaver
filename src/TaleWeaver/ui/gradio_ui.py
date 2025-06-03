import gradio as gr
from utils import constants
from core.server import Server
from loguru import logger 

async def create_gradio_interface(server: Server): # Accept server instance
    with gr.Blocks(theme=gr.themes.Soft(), title="TaleWeaver") as interface:
        gr.Markdown("# TaleWeaver")

        current_sessions = server.get_available_sessions()

        with gr.Tab("Setup"):
            with gr.Row():
                with gr.Column(scale=2):
                    session_name_input = gr.Textbox(label="New Session Name", placeholder="Enter a unique name for your game")
                    setting_input = gr.Textbox(label="Game Setting / World Description", lines=3, placeholder="e.g., A medieval fantasy kingdom on the brink of war")
                    language_dropdown = gr.Dropdown(
                        label="Select Language",
                        choices=list(constants.LANGUAGES.keys()),
                        value="english"
                    )
                    start_button = gr.Button("Start New Game", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### Add Player(s) to New Game")
                    player_name_input = gr.Textbox(label="Character Name")
                    player_backstory_input = gr.Textbox(label="Character Backstory", lines=2)
                    add_player_button = gr.Button("Add Player")
                    current_players_display = gr.Textbox(label="Players Added to New Game", interactive=False, lines=2, value="No players added yet.")

            with gr.Row():
                 with gr.Column():
                    load_dropdown = gr.Dropdown(
                        label="Load Previous Session",
                        choices=current_sessions,
                        value=current_sessions[0] if current_sessions else None,
                        interactive=True
                    )
                    load_button = gr.Button("Load Selected Session")

            setup_status = gr.Textbox(label="Setup Status", interactive=False, lines=2)

        with gr.Tab("Game"):
            chat_history_output = gr.Chatbot(label="Game History", height=600)
            with gr.Row():
                mode_dropdown = gr.Dropdown(
                    label="Action Mode",
                    choices=constants.MODES,
                    value="Say",
                    scale=1
                )
                action_input = gr.Textbox(
                    label="Your Action",
                    placeholder="Type what you say or do...",
                    scale=5,
                    show_label=False
                )
                submit_action_button = gr.Button("Submit Action", variant="primary", scale=1)

            story_button = gr.Button("Continue Story (AI Only)", scale=1)
            game_status_output = gr.Textbox(label="Game Status", value="No active game session.", interactive=False)

        with gr.Tab("Settings"):
            gr.Markdown("### Game Configuration")
            api_key_input = gr.Textbox(
                label="GROQ API Key",
                value=lambda: server.json_handler.get_setting('groq_api_key'),
                type="password"
            )
            model_input = gr.Textbox(
                label="AI Model",
                value=lambda: server.json_handler.get_setting('ai_settings.model'),
            )
            system_prompt_input = gr.Textbox(
                label="Base System Prompt for AI",
                value=lambda: server.json_handler.get_setting('ai_settings.system_prompt'),
                lines=5,
                max_lines=10
            )
            update_config_button = gr.Button("Update Game Configuration", variant="stop")
            config_status_output = gr.Textbox(label="Configuration Status", interactive=False)

        # --- Setup Tab Logic (Callbacks defined as nested functions to access 'server') ---
        async def add_player_gradio(name, backstory, current_player_list_str):
            if not name:
                return "Character name cannot be empty.", current_player_list_str
            status = await server.add_player(name, backstory)
            player_names = [p.name for p in server.players.values()]
            new_player_list_str = "\n".join(player_names) if player_names else "No players added yet."
            return status, new_player_list_str

        add_player_button.click(
            add_player_gradio,
            inputs=[player_name_input, player_backstory_input, current_players_display],
            outputs=[setup_status, current_players_display]
        ).then(lambda: (gr.Textbox(value=""), gr.Textbox(value="")), outputs=[player_name_input, player_backstory_input])

        async def start_game_gradio(session, setting, lang_key):
            if not session:
                return "Session name cannot be empty.", server.message_history, "Session name required.", gr.Dropdown(choices=server.get_available_sessions()), current_players_display.value
            if not server.players:
                 return "Please add at least one player before starting the game.", server.message_history, "Player(s) required.", gr.Dropdown(choices=server.get_available_sessions()), current_players_display.value

            lang_code = constants.LANGUAGES[lang_key]
            start_status = await server.start_game(session, setting, lang_code)
            updated_sessions = server.get_available_sessions()
            player_names_str = "\n".join([p.name for p in server.players.values()])
            # server.players.clear() # Decide if players should be cleared from server instance for next new game setup
                                     # For now, they persist on the server instance for the UI.
            # current_players_display_val = "No players added yet." # Reset UI display for new game players
            return f"{start_status}", [], f"Game '{session}' started. Language: {lang_key}", gr.Dropdown(choices=updated_sessions, value=session), player_names_str

        start_button.click(
            start_game_gradio,
            inputs=[session_name_input, setting_input, language_dropdown],
            outputs=[setup_status, chat_history_output, game_status_output, load_dropdown, current_players_display]
        )
        
        async def load_session_gradio(session_to_load, lang_key):
            if not session_to_load:
                 return "No session selected to load.", server.message_history, "Session selection required.", gr.Dropdown(choices=server.get_available_sessions()), "N/A"
            
            load_status = await server.load_session(session_to_load, lang_key)
            updated_sessions = server.get_available_sessions()
            player_names_str = "\n".join([p.name for p in server.players.values()]) if server.players else "No players in loaded session."
            # Update current_players_display with players from the loaded session
            return load_status, server.message_history, f"Loaded session: {session_to_load}. Language: {lang_key}", gr.Dropdown(choices=updated_sessions, value=session_to_load), player_names_str

        load_button.click(
            load_session_gradio,
            inputs=[load_dropdown, language_dropdown],
            outputs=[setup_status, chat_history_output, game_status_output, load_dropdown, current_players_display]
        )

        # --- Game Tab Logic ---
        async def chat_action_gradio_base(action_text, mode_choice, current_chat_history):
            if not server.current_session:
                system_message = ("System", "No active game session. Please start or load a game from the Setup tab.")
                return current_chat_history + [system_message] if system_message not in current_chat_history else current_chat_history, "", "Not in a game session."
            if not server.players:
                system_message = ("System", "No players in the current session. Please add players or load a session with players.")
                return current_chat_history + [system_message] if system_message not in current_chat_history else current_chat_history, "", "No player in session."

            player_name = list(server.players.keys())[0] # Simple single-player UI assumption
            new_messages = await server.action(action_text, player_name, mode_choice)
            updated_history = current_chat_history + new_messages
            return updated_history, "", f"Action processed for {player_name}."

        async def chat_action_gradio_submit(action_text, mode_choice, current_chat_history):
            return await chat_action_gradio_base(action_text, mode_choice, current_chat_history)

        async def chat_action_gradio_enter(action_text, mode_choice, current_chat_history):
            return await chat_action_gradio_base(action_text, mode_choice, current_chat_history)

        submit_action_button.click(
            chat_action_gradio_submit,
            inputs=[action_input, mode_dropdown, chat_history_output],
            outputs=[chat_history_output, action_input, game_status_output]
        )
        action_input.submit(
            chat_action_gradio_enter,
            inputs=[action_input, mode_dropdown, chat_history_output],
            outputs=[chat_history_output, action_input, game_status_output]
        )

        async def continue_story_gradio(current_chat_history):
            if not server.current_session:
                system_message = ("System", "No active game session. Please start or load a game.")
                return current_chat_history + [system_message] if system_message not in current_chat_history else current_chat_history, "Not in a game session."
            if not server.players:
                 system_message = ("System", "Cannot continue story without player context.")
                 return current_chat_history + [system_message] if system_message not in current_chat_history else current_chat_history, "No player context for story."

            player_name_context = list(server.players.keys())[0] if server.players else "Narrator"
            new_messages = await server.action("", player_name_context, "Story")
            updated_history = current_chat_history + new_messages
            return updated_history, "Story continued by AI."

        story_button.click(
            continue_story_gradio,
            inputs=[chat_history_output],
            outputs=[chat_history_output, game_status_output]
        )

        # --- Settings Tab Logic ---
        async def update_config_gradio(api_key_val, model_val, system_prompt_val):
            new_settings = {
                "groq_api_key": api_key_val,
                "ai_settings.model": model_val, # Corrected key for model
                "ai_settings.system_prompt": system_prompt_val # Corrected key for system_prompt
            }
            # Ensure keys match what JsonHandler expects (e.g., nested keys)
            update_status = await server.update_config(new_settings)
            return update_status

        update_config_button.click(
            update_config_gradio,
            inputs=[api_key_input, model_input, system_prompt_input],
            outputs=[config_status_output]
        )

        def refresh_sessions_and_settings_sync():
            sessions = server.get_available_sessions()
            # Settings values are updated via lambdas, so only choices for dropdowns need explicit refresh here.
            return gr.Dropdown(choices=sessions, value=sessions[0] if sessions else None)

        interface.load(refresh_sessions_and_settings_sync, outputs=[load_dropdown])

    return interface