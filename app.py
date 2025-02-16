import logging
import os
import re
from pathlib import Path
from uuid import uuid4

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(
    page_title="GroqFlow",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="auto",
)

try:
    for secret_name, secret_value in st.secrets.items():
        if isinstance(secret_value, (str, int, float, bool)):
            os.environ.setdefault(secret_name, str(secret_value))
except StreamlitSecretNotFoundError:
    pass

from groqflow.chat_service import generate_reply
from groqflow.chat_store import (
    append_turn,
    create_conversation,
    list_conversations,
    load_conversation,
)
from groqflow.config import PROJECT_ROOT, tenant_paths
from groqflow.config_loader import (
    BookingConfig,
    BookingField,
    TenantConfig,
    UIConfig,
    load_tenant_config,
    save_tenant_config,
)
from groqflow.resources import get_knowledge_base


def validate_runtime_setup() -> None:
    missing = []
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if missing:
        st.warning(
            "The LLM backend is not configured. "
            "Please provide a valid GROQ_API_KEY in your environment or .env configuration."
        )
        st.code("cp .env.example .env\n# Set GROQ_API_KEY in .env")
        st.stop()


validate_runtime_setup()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink: #182522; --muted: #586761; --line: #d5dfd9; --green: #147d64; --mint: #eaf5f0; }
    html, body, [data-testid="stAppViewContainer"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: #f5f8f6; color: var(--ink); }
    [data-testid="stAppViewContainer"] .main .block-container { max-width: 1240px; padding: 32px 32px 48px; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: 0; color: var(--ink) !important; }
    p, label, [data-testid="stProgress"] { color: var(--ink); }
    [data-testid="stSidebar"] { background: #eef3f0; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--ink); }
    .brand { font: 800 24px 'Manrope', sans-serif; color: var(--ink); padding: 4px 0 18px; }
    .eyebrow { color: var(--green); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .intro { color: var(--muted) !important; max-width: 720px; font-size: 16px; line-height: 1.6; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); border-radius: 8px; background: #fff; }
    .stButton button, .stFormSubmitButton button { border-radius: 6px; font-weight: 600; }
    .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] { background: var(--green); border-color: var(--green); }
    [data-testid="stChatMessage"] { border: 1px solid var(--line); border-radius: 8px; background: white; }
    [data-testid="stChatInput"] { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

CONFIG_DIR = PROJECT_ROOT / "configs"
config_files = sorted(CONFIG_DIR.glob("*.yaml")) + sorted(CONFIG_DIR.glob("*.yml"))
tenants: list[tuple[Path, TenantConfig]] = []
for config_file in config_files:
    try:
        tenants.append((config_file, load_tenant_config(str(config_file))))
    except Exception:
        logging.exception("Could not load tenant configuration %s", config_file)

if "owner_id" not in st.session_state:
    owner_id_path = PROJECT_ROOT / "data" / "local_owner_id"
    configured_owner_id = os.getenv("GROQFLOW_USER_ID")
    if configured_owner_id:
        st.session_state.owner_id = configured_owner_id
    else:
        owner_id_path.parent.mkdir(parents=True, exist_ok=True)
        if not owner_id_path.exists():
            owner_id_path.write_text(str(uuid4()), encoding="utf-8")
        st.session_state.owner_id = owner_id_path.read_text(encoding="utf-8").strip()
if "tenant_id" not in st.session_state:
    configured_config = Path(
        os.getenv("TENANT_CONFIG_PATH", "configs/demo_hospital.yaml")
    )
    if not configured_config.is_absolute():
        configured_config = PROJECT_ROOT / configured_config
    configured_tenant = next(
        (
            tenant
            for config_path, tenant in tenants
            if config_path.resolve() == configured_config.resolve()
        ),
        tenants[0][1] if tenants else None,
    )
    st.session_state.tenant_id = configured_tenant.id if configured_tenant else None
if "view" not in st.session_state:
    st.session_state.view = "workspace"
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "conversation_messages" not in st.session_state:
    st.session_state.conversation_messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {}

st.sidebar.markdown('<div class="brand">GroqFlow</div>', unsafe_allow_html=True)
tenant_options = {tenant.name: tenant for _, tenant in tenants}
tenant_names = list(tenant_options)
if not tenant_names:
    st.error("No valid chatbot configurations found. Create one to get started.")
    st.stop()

selected_tenant = next(
    (tenant for _, tenant in tenants if tenant.id == st.session_state.tenant_id),
    tenants[0][1] if tenants else None,
)
CREATE_CHATBOT_OPTION = "+ Create chatbot"
pending_workspace_choice = st.session_state.pop("pending_workspace_choice", None)
if pending_workspace_choice in tenant_options:
    st.session_state["workspace_selector"] = pending_workspace_choice
workspace_choice = st.sidebar.selectbox(
    "Chatbot",
    options=[*tenant_names, CREATE_CHATBOT_OPTION],
    index=(
        tenant_names.index(selected_tenant.name) if selected_tenant is not None else 0
    ),
    key="workspace_selector",
)

if workspace_choice == CREATE_CHATBOT_OPTION:
    st.markdown('<div class="eyebrow">New workspace</div>', unsafe_allow_html=True)
    st.title("Create a chatbot")
    with st.form("create_chatbot_form"):
        chatbot_name = st.text_input("Chatbot name", placeholder="Community Clinic")
        chatbot_description = st.text_area(
            "What should this chatbot represent?",
            placeholder="Describe the organization and the information this assistant should provide.",
        )
        booking_fields_text = st.text_input(
            "Required booking details",
            value="Full name, Phone number",
            help="Separate fields with commas. For example: Full name, Email, Phone number.",
        )
        create_chatbot = st.form_submit_button("Create chatbot", type="primary")

    if create_chatbot:
        cleaned_name = chatbot_name.strip()
        cleaned_description = chatbot_description.strip()
        field_labels = [
            label.strip() for label in booking_fields_text.split(",") if label.strip()
        ]
        if not cleaned_name or not cleaned_description or not field_labels:
            st.error(
                "Enter a chatbot name, description, and at least one booking field."
            )
        elif cleaned_name in tenant_options:
            st.error("A chatbot with that name already exists.")
        else:
            slug = re.sub(r"[^a-z0-9]+", "_", cleaned_name.lower()).strip("_")
            chatbot_id = f"custom_{slug or uuid4().hex[:8]}"
            config_path = CONFIG_DIR / f"{chatbot_id}.yaml"
            suffix = 2
            while config_path.exists():
                chatbot_id = f"custom_{slug}_{suffix}"
                config_path = CONFIG_DIR / f"{chatbot_id}.yaml"
                suffix += 1

            fields = []
            used_field_names: set[str] = set()
            for label in field_labels:
                field_name = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
                field_name = field_name or f"detail_{len(fields) + 1}"
                base_field_name = field_name
                suffix = 2
                while field_name in used_field_names:
                    field_name = f"{base_field_name}_{suffix}"
                    suffix += 1
                used_field_names.add(field_name)
                fields.append(
                    BookingField(name=field_name, description=label, required=True)
                )

            chatbot = TenantConfig(
                id=chatbot_id,
                name=cleaned_name,
                description=cleaned_description,
                ui=UIConfig(),
                booking=BookingConfig(fields=fields),
            )
            save_tenant_config(chatbot, config_path)
            st.session_state.tenant_id = chatbot.id
            st.session_state.view = "workspace"
            st.session_state.conversation_id = None
            st.session_state.conversation_messages = []
            st.session_state.agent_state = {}
            st.session_state.pending_workspace_choice = chatbot.name
            st.rerun()
    st.stop()

selected_tenant_id = tenant_options[workspace_choice].id
if selected_tenant_id != selected_tenant.id:
    st.session_state.tenant_id = selected_tenant_id
    st.session_state.view = "workspace"
    st.session_state.conversation_id = None
    st.session_state.conversation_messages = []
    st.session_state.agent_state = {}
    st.session_state.pop(f"history_{selected_tenant.id}", None)
    st.rerun()

selected_tenant = tenant_options[workspace_choice]
documents_path, vector_path = tenant_paths(selected_tenant)
documents_path.mkdir(parents=True, exist_ok=True)
pdf_files = sorted(
    path
    for path in documents_path.glob("*")
    if path.is_file()
    and not path.name.startswith(".")
    and path.suffix.lower() == ".pdf"
)
source_paths = [str(path.resolve()) for path in pdf_files]
source_key = f"active_sources_{selected_tenant.id}"
pending_sources_key = f"pending_sources_{selected_tenant.id}"
if pending_sources_key in st.session_state:
    st.session_state[source_key] = st.session_state.pop(pending_sources_key)
delete_sources_key = f"delete_sources_{selected_tenant.id}"
if st.session_state.pop(f"clear_delete_sources_{selected_tenant.id}", False):
    st.session_state[delete_sources_key] = []
if source_key not in st.session_state:
    st.session_state[source_key] = list(source_paths)
else:
    st.session_state[source_key] = [
        source for source in st.session_state[source_key] if source in source_paths
    ]
selected_sources = st.sidebar.multiselect(
    "Knowledge files",
    options=source_paths,
    format_func=lambda source: Path(source).name,
    key=source_key,
    help="Only selected files are used to answer knowledge questions.",
)
if not source_paths:
    st.sidebar.caption("Upload a PDF to add a knowledge source.")

if st.session_state.view == "workspace":
    st.markdown(
        '<div class="eyebrow">Knowledge workspace</div>', unsafe_allow_html=True
    )
    st.title(selected_tenant.name)
    st.markdown(
        "<p class='intro'>Choose how this assistant should use your organization’s knowledge. Add documents now, or continue with files already in the library.</p>",
        unsafe_allow_html=True,
    )
    st.progress(0.34, text="1 of 3 · Select a knowledge source")

    upload_column, library_column = st.columns([1.1, 0.9], gap="large")
    with upload_column:
        with st.container(border=True):
            st.subheader("Add documents")
            st.caption(
                "Upload a PDF or upload it again under the same name to replace its indexed version."
            )
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                help="PDF only. Maximum 25 MB per file.",
                key=f"uploads_{selected_tenant.id}",
            )
            if st.button(
                "Index uploaded documents",
                type="primary",
                disabled=not uploaded_files,
                use_container_width=True,
            ):
                total_bytes = sum(upload.size for upload in uploaded_files)
                if total_bytes > 100 * 1024 * 1024:
                    st.error("The combined upload must be under 100 MB.")
                elif any(upload.size > 25 * 1024 * 1024 for upload in uploaded_files):
                    st.error("Each PDF must be under 25 MB.")
                else:
                    knowledge_base = get_knowledge_base(selected_tenant)
                    indexed = 0
                    prepared_uploads = []
                    for upload in uploaded_files:
                        safe_name = Path(upload.name).name
                        destination = documents_path / safe_name
                        temporary = documents_path / f".{uuid4().hex}.upload.pdf"
                        temporary.write_bytes(upload.getvalue())
                        prepared_uploads.append((safe_name, temporary, destination))
                    try:
                        indexed_results = knowledge_base.update_many(
                            [
                                (str(temporary), str(destination))
                                for _, temporary, destination in prepared_uploads
                            ]
                        )
                        for (safe_name, temporary, destination), was_indexed in zip(
                            prepared_uploads, indexed_results
                        ):
                            if was_indexed:
                                temporary.replace(destination)
                                indexed += 1
                            else:
                                temporary.unlink(missing_ok=True)
                                st.warning(f"{safe_name} could not be read as a PDF.")
                    except Exception:
                        for _, temporary, _ in prepared_uploads:
                            temporary.unlink(missing_ok=True)
                        logging.exception("Failed to index uploaded documents")
                        st.error("Could not index the selected documents.")
                    if indexed:
                        newly_indexed_sources = [
                            str(destination.resolve())
                            for (_, _, destination), was_indexed in zip(
                                prepared_uploads, indexed_results
                            )
                            if was_indexed
                        ]
                        st.session_state[pending_sources_key] = list(
                            dict.fromkeys(
                                [*st.session_state[source_key], *newly_indexed_sources]
                            )
                        )
                        st.success(f"Indexed {indexed} document(s).")
                        st.rerun()

    with library_column:
        with st.container(border=True):
            st.subheader("Existing library")
            st.caption(
                "Continue with the documents already uploaded for this workspace."
            )
            if pdf_files:
                st.metric("Available PDFs", len(pdf_files))
                with st.expander("View documents", expanded=True):
                    for document in pdf_files:
                        st.write(f"PDF · {document.name}")
                delete_sources = st.multiselect(
                    "Delete files",
                    options=source_paths,
                    format_func=lambda source: Path(source).name,
                    key=delete_sources_key,
                )
                if st.button(
                    "Delete selected files",
                    disabled=not delete_sources,
                    use_container_width=True,
                ):
                    try:
                        get_knowledge_base(selected_tenant).delete_many(delete_sources)
                        for source in delete_sources:
                            Path(source).unlink(missing_ok=True)
                        st.session_state[pending_sources_key] = [
                            source
                            for source in st.session_state[source_key]
                            if source not in delete_sources
                        ]
                        st.session_state[
                            f"clear_delete_sources_{selected_tenant.id}"
                        ] = True
                        st.success(f"Deleted {len(delete_sources)} file(s).")
                        st.rerun()
                    except Exception:
                        logging.exception("Failed to delete knowledge files")
                        st.error("Could not delete the selected files.")
                if st.button(
                    "Use selected files",
                    disabled=not selected_sources,
                    use_container_width=True,
                ):
                    st.session_state.view = "chat"
                    st.session_state.conversation_id = create_conversation(
                        st.session_state.owner_id, selected_tenant.id
                    )
                    st.session_state[f"history_{selected_tenant.id}"] = (
                        st.session_state.conversation_id
                    )
                    st.session_state.conversation_messages = []
                    st.session_state.agent_state = {}
                    st.rerun()
            else:
                st.info("No PDFs are uploaded for this workspace yet.")

elif st.session_state.view == "chat":
    st.sidebar.divider()
    st.sidebar.subheader("Conversations")
    if st.sidebar.button("＋  New conversation", use_container_width=True):
        conversation_id = create_conversation(
            st.session_state.owner_id, selected_tenant.id
        )
        st.session_state.conversation_id = conversation_id
        st.session_state[f"history_{selected_tenant.id}"] = conversation_id
        st.session_state.conversation_messages = []
        st.session_state.agent_state = {}
        st.rerun()

    conversations = list_conversations(st.session_state.owner_id, selected_tenant.id)
    if conversations:
        conversation_ids = [item["id"] for item in conversations]
        current_id = st.session_state.conversation_id
        selected_id = st.sidebar.selectbox(
            "Chat history",
            options=conversation_ids,
            index=(
                conversation_ids.index(current_id)
                if current_id in conversation_ids
                else 0
            ),
            format_func=lambda item_id: next(
                item["title"] for item in conversations if item["id"] == item_id
            ),
            key=f"history_{selected_tenant.id}",
        )
        if selected_id != st.session_state.conversation_id:
            selected = load_conversation(
                selected_id, st.session_state.owner_id, selected_tenant.id
            )
            if selected:
                st.session_state.conversation_id = selected_id
                st.session_state.conversation_messages = selected["messages"]
                st.session_state.agent_state = selected["agent_state"]
                st.rerun()

    if st.sidebar.button("← Knowledge workspace", use_container_width=True):
        st.session_state.view = "workspace"
        st.rerun()

    st.markdown(
        '<div class="eyebrow">GroqFlow · Assistant</div>', unsafe_allow_html=True
    )
    st.title(selected_tenant.name)
    st.progress(1.0, text="3 of 3 · Conversation")

    if not st.session_state.conversation_id:
        st.info("Start a new conversation or choose one from chat history.")
    else:
        if not st.session_state.conversation_messages:
            with st.chat_message("assistant"):
                st.markdown(
                    f"Hello, I'm the {selected_tenant.name} assistant. How can I help?"
                )

        for message in st.session_state.conversation_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about your documents or services"):
            prior_messages = list(st.session_state.conversation_messages)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Searching and preparing a response..."):
                    try:
                        response, next_state = generate_reply(
                            selected_tenant,
                            prompt,
                            prior_messages,
                            st.session_state.agent_state,
                            source_filter=selected_sources,
                            conversation_id=st.session_state.conversation_id,
                        )
                    except Exception:
                        logging.exception("Chat response generation failed")
                        st.error(
                            "I could not complete that request. Check the model/API configuration and try again."
                        )
                        st.stop()
                st.markdown(response)
                if next_state.get("booking_confirmed"):
                    notification_status = next_state.get("booking_notification_status")
                    if notification_status == "sent":
                        st.success(
                            "Appointment request sent to the notification workflow."
                        )
                    elif notification_status == "pending":
                        st.warning(
                            "Request saved in this conversation; n8n notification is pending retry."
                        )
                    else:
                        st.info(
                            "Request saved in this conversation. n8n notifications are not configured."
                        )

            append_turn(
                st.session_state.conversation_id,
                st.session_state.owner_id,
                selected_tenant.id,
                prompt,
                response,
                next_state,
            )
            st.session_state.conversation_messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            )
            st.session_state.agent_state = next_state
            st.rerun()
