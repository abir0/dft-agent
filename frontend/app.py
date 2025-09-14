"""
A Streamlit app for interacting with the langgraph agent via a simple chat interface.
The app has three main functions which are all run async:

- main() - sets up the streamlit app and high level structure
- draw_messages() - draws a set of chat messages - either replaying existing messages
  or streaming new ones.
- handle_feedback() - Draws a feedback widget and records feedback from the user.

The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.

"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import asyncio
import base64
import os
import re
import urllib.parse
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from backend.agents.client import AgentClient, AgentClientError
from backend.core.schema import ChatHistory, ChatMessage

# Title and icon for head
APP_TITLE = "AI Assistant"
CAPTION = (
    "Multi-agent AI assistant for computational materials science and research."
)
SCRIPT_DIR = Path(__file__).parent
APP_ICON = SCRIPT_DIR / "static" / "logo.svg"
USER_ID_COOKIE = "user_id"


# Utility functions
def img_to_bytes(img_path: str | Path) -> str:
    """Convert image file to base64 string, with fallback for missing files."""
    try:
        img_path = Path(img_path)
        if not img_path.exists():
            # Return a simple placeholder SVG if file doesn't exist
            placeholder_svg = """<svg width="40" height="40" xmlns="http://www.w3.org/2000/svg">
                <rect width="40" height="40" fill="#f0f0f0" stroke="#ccc" stroke-width="1"/>
                <text x="20" y="25" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">AI</text>
            </svg>"""
            return base64.b64encode(placeholder_svg.encode()).decode()

        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        # Log the error but don't crash the app
        print(f"Warning: Could not load image {img_path}: {e}")
        # Return a simple placeholder
        placeholder_svg = """<svg width="40" height="40" xmlns="http://www.w3.org/2000/svg">
            <rect width="40" height="40" fill="#f0f0f0" stroke="#ccc" stroke-width="1"/>
            <text x="20" y="25" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">AI</text>
        </svg>"""
        return base64.b64encode(placeholder_svg.encode()).decode()


def img_to_html(img_path: str | Path) -> str:
    """Convert image file to HTML img tag with base64 data."""
    try:
        img_path = Path(img_path)
        if not img_path.exists():
            return "<div style='width:40px;height:40px;background:#f0f0f0;border:1px solid #ccc;display:flex;align-items:center;justify-content:center;font-size:12px;color:#666;'>AI</div>"

        img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
            img_to_bytes(img_path)
        )
        return img_html
    except Exception as e:
        print(f"Warning: Could not create HTML for image {img_path}: {e}")
        return "<div style='width:40px;height:40px;background:#f0f0f0;border:1px solid #ccc;display:flex;align-items:center;justify-content:center;font-size:12px;color:#666;'>AI</div>"


def replace_img_tag(html_content: str) -> str:
    def replacer(match):
        img_path = match.group(1)
        if Path(img_path).exists():
            html = img_to_html(img_path)
            return html
        else:
            return

    return re.sub(
        r"<img\s+[^>]*src=[\'\"]([^\'\"]+)[\'\"][^>]*>",
        replacer,
        html_content,
        flags=re.IGNORECASE,
    )


def get_or_create_user_id() -> str:
    """Retrieve or create a persistent (per-URL) user id.

    Mirrors logic from reference app: prefer session_state, else query params, else new uuid.
    Adds it back into URL params for share/resume convenience.
    """
    # Existing in session
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Provided in URL
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Create new
    user_id = str(uuid.uuid4())
    st.session_state[USER_ID_COOKIE] = user_id
    st.query_params[USER_ID_COOKIE] = user_id
    return user_id


def safe_status(label: str, **kwargs):
    """Wrapper around st.status that strips unsupported surrogate pairs.

    If a UnicodeEncodeError occurs, fall back to ASCII friendly label.
    """
    try:
        return st.status(label, **kwargs)
    except UnicodeEncodeError:
        safe_label = label.encode("utf-8", "ignore").decode("utf-8")
        # Remove any leftover lone surrogates just in case
        safe_label = safe_label.encode("utf-16", "surrogatepass").decode(
            "utf-16", "ignore"
        )
        # Fallback: strip emoji entirely if still problematic
        safe_label = re.sub(r"[\u2600-\u27BF\U0001F000-\U0001FAFF]", "", safe_label)
        return st.status(safe_label, **kwargs)


async def main() -> None:
    # Set page icon only if it exists
    page_icon = APP_ICON if APP_ICON.exists() else None

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=page_icon,
        menu_items={},
    )

    # Hide streamlit upper-right status and deploy buttons
    st.html(
        """
        <style>
            [data-testid="stStatusWidget"],
            [data-testid="stAppDeployButton"] {
                visibility: hidden;
                height: 0;
                position: fixed;
            }
        </style>
        """,
    )
    # # Hide the streamlit toolbar
    # if st.get_option("client.toolbarMode") != "minimal":
    #     st.set_option("client.toolbarMode", "minimal")
    #     await asyncio.sleep(0.1)
    #     st.rerun()

    # Obtain / persist a user id (not currently sent to backend, reserved for future use)
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8083)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            # Switch to uuid for shareable / multi-session separation
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(
                    thread_id=thread_id
                ).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Config options
    with st.sidebar:
        # Header w/ icon
        st.markdown(
            f"""
            <h1 style=\"display: flex; align-items: center;\">
                <img src=\"data:image/svg+xml;base64,{img_to_bytes(APP_ICON)}\" width=\"40\" style=\"margin-right: 10px;\">
                {APP_TITLE}
            </h1>
            """,
            unsafe_allow_html=True,
        )
        st.caption(CAPTION)

        # New Chat button
        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            # Convert models to strings for display
            model_options = [str(m) for m in agent_client.info.models]
            default_model_str = str(agent_client.info.default_model)
            model_idx = (
                model_options.index(default_model_str)
                if default_model_str in model_options
                else 0
            )
            model = st.selectbox("LLM to use", options=model_options, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use", options=agent_list, index=agent_idx
            )
            use_streaming = st.toggle("Stream results", value=True)
            # Display user id (read-only)
            st.text_input("User ID", value=user_id, disabled=True)

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded for evaluation and improvement purposes."
            )

        @st.dialog("Share Chat")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [
                    session.client.request.protocol,
                    session.client.request.host,
                    "",
                    "",
                    "",
                    "",
                ]
            )
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or resume this chat")

        if st.button(":material/upload: Share Chat", use_container_width=True):
            share_chat_dialog()

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        # Basic agent-specific welcome variants (extendable)
        match agent_client.agent:
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "dft_agent":
                WELCOME = "Hi! I'm the DFT workflow agent. Provide a materials/DFT request to begin."
            case _:
                WELCOME = "Hello! I'm an AI-powered chat assistant. How can I help you?"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                )  # user_id reserved (backend not expecting yet)
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                )
                messages.append(response)
                st.chat_message("ai").write(response.content)
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        # Normalize message type to handle any case sensitivity issues
        msg_type = (
            msg.type.strip().lower()
            if isinstance(msg.type, str)
            else str(msg.type).strip().lower()
        )

        match msg_type:
            # Messages from the user
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # Messages from the agent with streaming tokens and tool calls
            case "ai":
                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # Reset the streaming variables to prepare for the next message
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(
                                replace_img_tag(msg.content), unsafe_allow_html=True
                            )
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(replace_img_tag(msg.content), unsafe_allow_html=True)

                    if msg.tool_calls:
                        # Detect if this is a delegated / transfer style tool call
                        if any(
                            "transfer_to" in tc.get("name", "") for tc in msg.tool_calls
                        ):
                            # Handle nested / delegated agent session
                            for tc in msg.tool_calls:
                                if "transfer_to" in tc.get("name", ""):
                                    status = safe_status(
                                        f"🤖 Sub Agent: {tc['name']}",
                                        state="running" if is_new else "complete",
                                        expanded=True,
                                    )
                                    await handle_sub_agent_msgs(
                                        messages_agen, status, is_new
                                    )
                        else:
                            # Standard tool call rendering
                            call_results = {}
                            for tool_call in msg.tool_calls:
                                status = safe_status(
                                    f"🛠️ Tool Call: {tool_call['name']}",
                                    state="running" if is_new else "complete",
                                )
                                call_results[tool_call["id"]] = status
                                status.write("Input:")
                                status.write(tool_call["args"])

                            # Expect one ToolMessage for each tool call
                            for _ in range(len(call_results)):
                                tool_result: ChatMessage = await anext(messages_agen)
                                if tool_result.type != "tool":
                                    st.error(
                                        f"Unexpected ChatMessage type: {tool_result.type}"
                                    )
                                    st.write(tool_result)
                                    st.stop()

                                if is_new:
                                    st.session_state.messages.append(tool_result)
                                    status = call_results.get(tool_result.tool_call_id)
                                if status:
                                    status.write("Output:")
                                    status.write(tool_result.content)
                                    status.update(state="complete")
            case "custom":
                # Future custom message handling placeholder
                if is_new:
                    st.session_state.messages.append(msg)
                if last_message_type != "custom":
                    last_message_type = "custom"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    if msg.content:
                        st.write(msg.content)

            # Handle tool messages that might come through the main loop
            case "tool":
                if is_new:
                    st.session_state.messages.append(msg)

                # Display tool results in the current AI message container
                if st.session_state.last_message:
                    with st.session_state.last_message:
                        st.write(f"**Tool Result:** {msg.content}")
                else:
                    # If no AI message container exists, create one
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                    with st.session_state.last_message:
                        st.write(f"**Tool Result:** {msg.content}")

            # For unexpected message types, log an error and stop
            case _:
                if hasattr(msg, "type"):
                    st.error(f"Unexpected ChatMessage type: {msg.type}")
                else:
                    st.error(f"Unexpected message format: {type(msg)}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if (
        feedback is not None
        and (latest_run_id, feedback) != st.session_state.last_feedback
    ):
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    """Handle nested delegated agent tool call message sequences.

    Reads subsequent messages until a *transfer_back_to* tool call completes.
    """
    nested_popovers = {}

    first_msg = await anext(messages_agen)
    if first_msg and is_new:
        st.session_state.messages.append(first_msg)

    while True:
        sub_msg = await anext(messages_agen)
        if sub_msg is None:
            break
        if is_new and hasattr(sub_msg, "type"):
            st.session_state.messages.append(sub_msg)

        # Tool result mapping for previously opened popover
        if (
            getattr(sub_msg, "type", None) == "tool"
            and getattr(sub_msg, "tool_call_id", None) in nested_popovers
        ):
            pop = nested_popovers[sub_msg.tool_call_id]
            pop.write("**Output:**")
            pop.write(sub_msg.content)
            continue

        # Completion condition: transfer back
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    transfer_result = await anext(messages_agen)
                    if transfer_result and is_new:
                        st.session_state.messages.append(transfer_result)
            status.update(state="complete")
            break

        # Regular content
        if status and getattr(sub_msg, "content", None):
            status.write(sub_msg.content)

        # Nested tool calls inside delegated agent
        if status and hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
            for tc in sub_msg.tool_calls:
                if "transfer_to" in tc.get("name", ""):
                    nested_status = safe_status(
                        f"🤖 Sub Agent: {tc['name']}",
                        state="running" if is_new else "complete",
                        expanded=True,
                    )
                    await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                else:
                    popover = status.popover(f"{tc['name']}", icon="🛠️")
                    popover.write(f"**Tool:** {tc['name']}")
                    popover.write("**Input:**")
                    popover.write(tc.get("args", {}))
                    nested_popovers[tc.get("id")] = popover


if __name__ == "__main__":
    asyncio.run(main())
