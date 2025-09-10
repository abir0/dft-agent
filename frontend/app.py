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
from collections.abc import AsyncGenerator
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx
import copy
from plan_helpers import validate_plan_schema, diff_plans
from agents.client import AgentClient, AgentClientError
from backend.core.schema import ChatHistory, ChatMessage
import json
import requests

# Title and icon for head
APP_TITLE = "AI Agent Interface"
APP_DIR = Path(__file__).parent
APP_ICON =APP_DIR/"static"/"logo.svg"

def fetch_plan(base_url: str, req: str, hints: dict | None = None, code: dict | None = None,history: list | None = None) -> dict:
    """Call the backend planner and return {request, plan, workroot}."""
    url = f"{base_url.rstrip('/')}/dft-planner/plan"
    payload = {"request": req, "hints": hints or {}, "code": code or {}, "history": history or []}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

# Utility functions
def img_to_bytes(img_path: str) -> bytes:
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def img_to_html(img_path: str) -> str:
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
        img_to_bytes(img_path)
    )
    return img_html


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

def plan_to_markdown(plan: dict) -> str:
    """Turn planner JSON into a readable outline."""
    if not isinstance(plan, dict):
        return "*(Invalid plan format)*"

    out = []
    goal = plan.get("goal")
    if goal:
        out.append(f"**Goal:** {goal}\n")

    assumptions = plan.get("assumptions") or []
    if assumptions:
        out.append("**Assumptions**")
        for a in assumptions:
            out.append(f"- {a}")
        out.append("")

    inputs = plan.get("inputs_summary") or {}
    if inputs:
        out.append("**Inputs**")
        for k, v in inputs.items():
            out.append(f"- **{k}**: {v}")
        out.append("")

    steps = plan.get("steps") or []
    if steps:
        out.append("**Plan**")
        for i, s in enumerate(steps, 1):
            tool = s.get("tool", "<?>")
            args = s.get("args", {})
            explain = s.get("explain", "")
            # concise args rendering
            arg_pairs = ", ".join(f"{k}={v}" for k, v in args.items())
            line = f"{i}. **{tool}**({arg_pairs})"
            if explain:
                line += f" — {explain}"
            out.append(line)
        out.append("")

    artifacts = plan.get("artifacts") or []
    if artifacts:
        out.append("**Artifacts**")
        for a in artifacts:
            out.append(f"- {a}")
        out.append("")

    crit = plan.get("success_criteria") or []
    if crit:
        out.append("**Success criteria**")
        for c in crit:
            out.append(f"- {c}")

    return "\n".join(out).strip()


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
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

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
                st.session_state.agent_url = agent_url
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
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

        if "last_plan" not in st.session_state:
            st.session_state.last_plan = None
        if "edited_plan" not in st.session_state:
            st.session_state.edited_plan = None

    # Config options
    with st.sidebar:
        # Add header with inline icon image
        st.markdown(
            f"""
            <h1 style="display: flex; align-items: center;">
                <img src="data:image/svg+xml;base64,{img_to_bytes(APP_ICON)}" width="40" style="margin-right: 10px;">
                  {APP_TITLE}
            </h1>
            """,
            unsafe_allow_html=True,
        )
        # Description
        st.markdown(
            """
            AI agent built with LangGraph, FastAPI and Streamlit
            """
        )
        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox(
                "LLM to use", options=agent_client.info.models, index=model_idx
            )
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("Stream results", value=True)
        st.markdown("----")
        use_planner = st.toggle(
            "Planner mode",
            value = False,
            help="When on, your message goes to /dft-planner/plan and the JSON plan is shown.")
        code_choice = st.selectbox("DFT code for planning", ["qe", "vasp"], index=0)
        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded for evaluation and improvement purposes."
            )

        @st.dialog("Share chat")
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
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}"
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or resume this chat")

        if st.button(":material/upload: Share chat", use_container_width=True):
            share_chat_dialog()

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
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
            if use_planner:
                RECENCY_WINDOW = 6
                base_url = st.session_state.get("agent_url") or os.getenv("AGENT_URL") or "http://localhost:8080"
                history_dicts = [msg.model_dump() for msg in st.session_state.messages]
                relevant_history= history_dicts[-RECENCY_WINDOW:]
                result = fetch_plan(base_url, user_input, hints=None, code={"code": code_choice},
                                    history = relevant_history)
                plan = result.get("plan", {})
                workroot = result.get("workroot", "")
                # keep originals for diff / reset
                st.session_state.last_plan = copy.deepcopy(plan)
                st.session_state.edited_plan = None

                pretty = plan_to_markdown(plan)

                plan_md = "**Plan generated**  \n"
                if workroot:
                    plan_md += f"_Workspace_: `{workroot}`\n\n"
                plan_md += pretty

                plan_msg = ChatMessage(type="ai", content=plan_md)
                messages.append(plan_msg)
                st.rerun()
            else:
                if use_streaming:
                    stream = agent_client.astream(
                        message=user_input,
                        model=model,
                        thread_id=st.session_state.thread_id,
                    )
                    await draw_messages(stream, is_new=True)
                else:
                    response = await agent_client.ainvoke(
                        message=user_input,
                        model=model,
                        thread_id=st.session_state.thread_id,
                    )
                    messages.append(response)
                    st.chat_message("ai").write(response.content)
                st.rerun()
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Planner error: {e}")
            st.stop()

    if st.session_state.get("last_plan"):
        if len(messages) > 0 and messages[-1].type == "ai":
            with st.chat_message("ai"):
                plan_to_show = st.session_state.get("edited_plan") or st.session_state.get("last_plan")
                st.markdown(plan_to_markdown(plan_to_show))
            with st.expander("Edit plan JSON", expanded=True):  #
                default_text = json.dumps(plan_to_show, indent=2)  #
                plan_text = st.text_area(  #
                    "Tweak and validate",  #
                    default_text,  #
                    key=f"edit_plan_{len(messages)}",  #
                    height=340,  #
                )  #
                cols = st.columns([1, 1, 1, 2])  #
                with cols[0]:  #
                    # --- This button logic now contains the complete fix ---
                    if st.button("Validate edits"):  #
                        try:  #
                            candidate = json.loads(plan_text)  #
                            errs = validate_plan_schema(candidate)  #
                            if errs:  #
                                st.error("Found issues:\n- " + "\n- ".join(errs))  #
                            else:  #
                                st.session_state.edited_plan = candidate  #
                                diffs = diff_plans(st.session_state.last_plan, candidate)  #
                                if diffs:  #
                                    st.success("Valid JSON Changes:\n- " + "\n- ".join(diffs))  #
                                else:  #
                                    st.success("Valid JSON (no changes)")  #

                                # THE FIX: Update message content and rerun to show the changes
                                new_plan_md = plan_to_markdown(candidate)
                                st.session_state.messages[-1].content = new_plan_md
                                st.rerun()

                        except Exception as e:  #
                            st.error(f"Invalid JSON: {e}")  #

                with cols[1]:  #
                    if st.button("Reset edits"):  #
                        st.session_state.edited_plan = None  #
                        # Also reset the message content to the original plan
                        original_plan_md = plan_to_markdown(st.session_state.last_plan)
                        st.session_state.messages[-1].content = original_plan_md
                        st.rerun()  #

                with cols[2]:  #
                    download_obj = st.session_state.edited_plan or st.session_state.last_plan  #
                    st.download_button(  #
                        "Download JSON",  #
                        data=json.dumps(download_obj or {}, indent=2),  #
                        file_name="dft_plan.json",  #
                        mime="application/json",  #
                    )  #

            # Dry run (no backend execution) to show step list and catch issues
            with st.expander("Dry run (no execution)"):  #
                plan_use = st.session_state.edited_plan or st.session_state.last_plan  #
                problems = validate_plan_schema(plan_use)  #
                if problems:  #
                    st.error("Please fix these before running:\n\n- " + "\n- ".join(problems))  #
                else:  #
                    st.success("Plan looks consistent")  #
                    for idx, step in enumerate(plan_use.get("steps", []), 1):  #
                        tool = step.get("tool")  #
                        args = step.get("args", {})  #
                        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())  #
                        st.write(f"{idx}. **{tool}**({arg_str})")  #
                    st.caption("Simulation only—no DFT jobs submitted.")

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
        match msg.type:
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
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
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
                            status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            # For unexpected message types, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
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


if __name__ == "__main__":
    asyncio.run(main())
