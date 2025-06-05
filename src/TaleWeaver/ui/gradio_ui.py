import gradio as gr
from utils import constants
from core.server import Server
from loguru import logger
import json 

def _add_system_message(history, text, tag="System"):
    msg = (None, f"{tag}: {text}")
    if history is None: history = []
    if not history or history[-1] != msg: history.append(msg)
    return history

# --- Helper functions for UI structure ---
def _build_setup_tab_components(server: Server):
    session_name_input = gr.Textbox(label="New Session Name", placeholder="e.g., MyEpicAdventure")
    setting_input = gr.Textbox(label="Game Setting / World Description", lines=3, placeholder="e.g., A medieval fantasy kingdom...")
    language_dropdown = gr.Dropdown(label="Select Language", choices=list(constants.LANGUAGES.keys()), value="english")
    start_button = gr.Button("Start New Game", variant="primary")
    
    current_sessions = server.get_available_sessions()
    load_dropdown = gr.Dropdown(label="Load Previous Session", choices=current_sessions or [], value=current_sessions[0] if current_sessions else None, interactive=True)
    load_button = gr.Button("Load Selected Session")
    setup_status_output = gr.Textbox(label="Setup Status", interactive=False, lines=1)
    return session_name_input, setting_input, language_dropdown, start_button, load_dropdown, load_button, setup_status_output

def _build_game_tab_components():
    chat_history_output = gr.Chatbot(label="Game History", height=600, bubble_full_width=False, avatar_images=(None, "assets/ai_avatar.png"))
    action_input = gr.Textbox(label="Your Action", placeholder="Type your action...", scale=5, show_label=False)
    submit_action_button = gr.Button("Submit Action", variant="primary", scale=1)
    story_button = gr.Button("Continue Story (AI Only)", scale=1)
    game_status_output = gr.Textbox(label="Game Status", value="No active game session.", interactive=False, lines=1)
    return chat_history_output, action_input, submit_action_button, story_button, game_status_output

def _build_character_config_tab_components(server: Server):
    character_name_input = gr.Textbox(label="Character Name", value=lambda: server.character.name)
    character_backstory_input = gr.Textbox(label="Character Backstory", lines=5, value=lambda: server.character.backstory)
    update_character_button = gr.Button("Save Character Details", variant="primary")
    character_config_status_output = gr.Textbox(label="Status", interactive=False, lines=1)
    return character_name_input, character_backstory_input, update_character_button, character_config_status_output

def _build_ai_config_tab_components(server: Server):
    active_provider_dropdown = gr.Dropdown(
        label="Active AI Provider",
        choices=list(server.json_handler.get_setting('ai_settings.providers', {}).keys()) or ['groq'],
        value=lambda: server.json_handler.get_setting('ai_settings.active_provider', 'groq'),
        interactive=True
    )
    provider_api_key_input = gr.Textbox(label="API Key", type="password", interactive=True)
    provider_model_input = gr.Textbox(label="Model Name", interactive=True)
    provider_base_url_input = gr.Textbox(label="Base URL", interactive=True)
    provider_additional_params_input = gr.Textbox(
        label="Additional Params (JSON string)", lines=3, interactive=True, placeholder='e.g., {"temperature": 0.7}'
    )
    system_prompt_input = gr.Textbox(
        label="Base System Prompt",
        value=lambda: server.json_handler.get_setting('ai_settings.system_prompt', ''),
        lines=8, interactive=True
    )
    update_ai_config_button = gr.Button("Update AI Configuration", variant="stop")
    ai_config_status_output = gr.Textbox(label="Status", interactive=False, lines=1)
    return (active_provider_dropdown, provider_api_key_input, provider_model_input,
            provider_base_url_input, provider_additional_params_input, system_prompt_input,
            update_ai_config_button, ai_config_status_output)

async def create_gradio_interface(server: Server):
    with gr.Blocks(theme=gr.themes.Soft(), title="TaleWeaver") as interface:
        gr.Markdown("# TaleWeaver")

        # --- Tab specific component definitions ---
        with gr.Tab("Setup"):
            (session_name_input, setting_input, language_dropdown, start_button, 
             load_dropdown, load_button, setup_status_output) = _build_setup_tab_components(server)

        with gr.Tab("Game"):
            (chat_history_output, action_input, submit_action_button, 
             story_button, game_status_output) = _build_game_tab_components()

        with gr.Tab("Settings"):
            with gr.Tabs():
                with gr.TabItem("Character Configuration"):
                    (character_name_input, character_backstory_input, update_character_button, 
                     character_config_status_output) = _build_character_config_tab_components(server)
                with gr.TabItem("AI Provider & Model Configuration"):
                    (active_provider_dropdown, provider_api_key_input, provider_model_input,
                     provider_base_url_input, provider_additional_params_input, system_prompt_input,
                     update_ai_config_button, ai_config_status_output) = _build_ai_config_tab_components(server)

        # --- Dynamic UI update for provider settings ---
        def get_provider_config_value(provider_key, setting_key, default=''):
            return server.json_handler.get_setting(f'ai_settings.providers.{provider_key}.{setting_key}', default)
        def get_provider_additional_params_str(provider_key):
            params = server.json_handler.get_setting(f'ai_settings.providers.{provider_key}.additional_params', {})
            return json.dumps(params, indent=2) if params else "{}"

        def update_provider_fields_display(selected_provider):
            return (get_provider_config_value(selected_provider, 'api_key'),
                    get_provider_config_value(selected_provider, 'model'),
                    get_provider_config_value(selected_provider, 'base_url'),
                    get_provider_additional_params_str(selected_provider))
        
        active_provider_dropdown.change(
            fn=update_provider_fields_display, inputs=[active_provider_dropdown],
            outputs=[provider_api_key_input, provider_model_input, provider_base_url_input, provider_additional_params_input]
        )
        
        # --- Callbacks ---
        async def start_game_cb(session, setting, lang_key):
            if not session.strip(): return "Session name required.", [], "Session name cannot be empty.", gr.update()
            status = await server.start_game(session, setting, constants.LANGUAGES.get(lang_key, "en"))
            new_history = server.get_current_message_history()
            game_status = f"Game '{session}' started. Lang: {lang_key}." if "started" in status.lower() else status
            return status, new_history, game_status, gr.update(choices=server.get_available_sessions(), value=session)

        start_button.click(
            start_game_cb, [session_name_input, setting_input, language_dropdown],
            [setup_status_output, chat_history_output, game_status_output, load_dropdown]
        ).then(lambda: (gr.update(value=""), gr.update(value="")), outputs=[session_name_input, setting_input])

        async def load_session_cb(session_to_load, lang_key):
            if not session_to_load: return "No session selected.", [], "Select session.", gr.update(value=None)
            status, history = await server.load_session(session_to_load, lang_key)
            game_status = f"Loaded '{session_to_load}'. Lang: {lang_key}." if "loaded" in status.lower() else status
            return status, history, game_status, gr.update(value=session_to_load)

        load_button.click(
            load_session_cb, [load_dropdown, language_dropdown],
            [setup_status_output, chat_history_output, game_status_output, load_dropdown]
        )

        async def handle_chat_action_cb(action_text, current_ui_history):
            if not server.session_manager.is_active():
                msg = server.json_handler.get_setting("user_messages.error_not_in_game_session", "No active game.")
                return _add_system_message(current_ui_history, msg), "", "Not in game."
            if not action_text.strip():
                 return current_ui_history, gr.update(value=""), "Please type an action."

            new_messages = await server.action(action_text, "PlayerInput")
            return current_ui_history + new_messages, gr.update(value=""), f"Action by {server.character.name} processed."

        submit_action_button.click(handle_chat_action_cb, [action_input, chat_history_output], [chat_history_output, action_input, game_status_output])
        action_input.submit(handle_chat_action_cb, [action_input, chat_history_output], [chat_history_output, action_input, game_status_output])

        async def continue_story_cb(current_ui_history):
            if not server.session_manager.is_active():
                msg = server.json_handler.get_setting("user_messages.error_not_in_game_session", "No active game.")
                return _add_system_message(current_ui_history, msg), "Not in game."
            new_messages = await server.action("", "Story")
            return current_ui_history + new_messages, "Story continued."

        story_button.click(continue_story_cb, [chat_history_output], [chat_history_output, game_status_output])

        async def update_char_config_cb(name, backstory):
            success, msg = await server.update_character_config(name, backstory)
            return msg, server.character.name, server.character.backstory

        update_character_button.click(update_char_config_cb, [character_name_input, character_backstory_input],
                                      [character_config_status_output, character_name_input, character_backstory_input])

        async def update_ai_config_cb(provider, api_key, model, base_url, params_str, sys_prompt):
            settings_to_update = {
                "ai_settings.active_provider": provider,
                f"ai_settings.providers.{provider}.api_key": api_key,
                f"ai_settings.providers.{provider}.model": model,
                f"ai_settings.providers.{provider}.base_url": base_url,
                "ai_settings.system_prompt": sys_prompt
            }
            try:
                params_dict = json.loads(params_str) if params_str.strip() else {}
                if not isinstance(params_dict, dict): raise ValueError("Must be JSON object")
                settings_to_update[f"ai_settings.providers.{provider}.additional_params"] = params_dict
            except Exception as e:
                return f"Error in Additional Params JSON: {e}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            status_msg = await server.update_config(settings_to_update)
            
            # Fetch fresh values from config to update UI
            new_provider = server.json_handler.get_setting('ai_settings.active_provider')
            api_k, mod, burl, add_p = update_provider_fields_display(new_provider)
            new_sys_prompt = server.json_handler.get_setting('ai_settings.system_prompt', '')

            return (status_msg, gr.update(value=new_provider, choices=list(server.json_handler.get_setting('ai_settings.providers', {}).keys())),
                    gr.update(value=api_k), gr.update(value=mod), gr.update(value=burl),
                    gr.update(value=add_p), gr.update(value=new_sys_prompt))

        update_ai_config_button.click(
            update_ai_config_cb,
            [active_provider_dropdown, provider_api_key_input, provider_model_input, provider_base_url_input, provider_additional_params_input, system_prompt_input],
            [ai_config_status_output, active_provider_dropdown, provider_api_key_input, provider_model_input, provider_base_url_input, provider_additional_params_input, system_prompt_input]
        )

        def refresh_ui_on_load_cb():
            sessions = server.get_available_sessions()
            current_provider = server.json_handler.get_setting('ai_settings.active_provider', 'groq')
            api_key, model, base_url, additional_params = update_provider_fields_display(current_provider)
            return (
                gr.update(choices=sessions, value=sessions[0] if sessions else None),
                gr.update(value=server.character.name), gr.update(value=server.character.backstory),
                gr.update(value=current_provider, choices=list(server.json_handler.get_setting('ai_settings.providers', {}).keys())),
                gr.update(value=api_key), gr.update(value=model), gr.update(value=base_url), gr.update(value=additional_params),
                gr.update(value=server.json_handler.get_setting('ai_settings.system_prompt', ''))
            )

        interface.load(
            refresh_ui_on_load_cb, 
            outputs=[load_dropdown, character_name_input, character_backstory_input,
                     active_provider_dropdown, provider_api_key_input, provider_model_input,
                     provider_base_url_input, provider_additional_params_input, system_prompt_input]
        )
    return interface