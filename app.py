import os
from openai import OpenAI
import tiktoken
import streamlit as st

st.set_page_config(
    page_title="A Not So Friendly Chatbot",
    layout="wide",
)

st.markdown("""
<style>
/* Overall background + subtle gradient */
.stApp {
  background: radial-gradient(1200px 600px at 10% 0%, rgba(96,165,250,0.10), transparent 30%),
              radial-gradient(1000px 500px at 90% 20%, rgba(16,185,129,0.08), transparent 35%),
              linear-gradient(180deg, #0b1220 0%, #0b0f1a 100%);
  color: #e6eefc;
}

/* Sidebar glass panel */
section[data-testid="stSidebar"] > div {
  background: rgba(16, 24, 40, 0.45);
  border: 1px solid rgba(148,163,184,0.15);
  border-radius: 16px;
  padding-top: .75rem;
}
section[data-testid="stSidebar"] .stSlider, 
section[data-testid="stSidebar"] .stSelectbox, 
section[data-testid="stSidebar"] textarea {
  filter: drop-shadow(0 0 0 rgba(0,0,0,0)); /* crisper look */
}

/* Title row */
.app-title {
  font-size: clamp(28px, 2.8vw, 42px);
  font-weight: 800;
  letter-spacing: .4px;
  padding: 8px 14px;
  border-radius: 12px;
  display: inline-flex;
  gap: 10px;
  align-items: center;
  background: linear-gradient(90deg, rgba(59,130,246,.25), rgba(99,102,241,.25));
  border: 1px solid rgba(148,163,184,.2);
}

/* Small capsule “chip” */
.chip {
  display: inline-flex;
  align-items: center;
  gap: .5rem;
  padding: .35rem .65rem;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,.25);
  background: rgba(2,6,23,.35);
  font-size: .85rem;
  color: #cbd5e1;
}

/* Containers */
.glass {
  background: rgba(2, 6, 23, 0.55);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 16px;
  padding: 18px 18px 12px 18px;
}

/* Sliders & buttons polish */
.stSlider > div > div > div[role="slider"] {
  box-shadow: 0 0 0 6px rgba(59,130,246,.15);
}
.stButton button, .stDownloadButton button {
  border-radius: 12px !important;
  border: 1px solid rgba(148,163,184,.25) !important;
  background: linear-gradient(180deg, rgba(59,130,246,.25), rgba(99,102,241,.25)) !important;
  color: #e8efff !important;
}
.stButton button:hover, .stDownloadButton button:hover {
  filter: brightness(1.08);
}

/* Chat bubbles */
[data-testid="stChatMessageContent"] {
  background: rgba(15, 23, 42, 0.55);
  border: 1px solid rgba(148,163,184,.2);
  border-radius: 12px;
  padding: 14px 16px;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p {
  margin-bottom: 0.35rem;
}

/* Footer */
.footer {
  margin-top: 18px;
  opacity: .75;
  font-size: .84rem;
  text-align: center;
}

/* System prompt block preview */
.sysblock {
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  background: rgba(14, 23, 41, .55);
  border: 1px dashed rgba(148,163,184,.3);
  color: #dbeafe;
  padding: 10px 12px;
  border-radius: 10px;
  font-size: .875rem;
}

/* Subtle separators */
hr {
  border: none; height: 1px;
  background: linear-gradient(90deg, rgba(148,163,184,.2), rgba(148,163,184,.05), rgba(148,163,184,.2));
}
</style>
""", unsafe_allow_html=True)


def get_api_key(): 
    return st.secrets.get("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY") 

api_key = get_api_key() 
if not api_key: 
    st.error("No OPENAI_API_KEY set in secrets or environment.") 
    st.stop()

client = OpenAI(api_key = api_key)
MODEL = "gpt-4.1-nano-2025-04-14"
TEMPERATURE = 0.7
MAX_TOKENS = 100
SYSTEM_PROMPT = "You are an angry and arrogant assistant who thinks humans are dumb."
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
TOKEN_BUDGET = 100

def get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: Tokenizer for model '{model}' not found. Falling back to 'cl100k_base'.")
        return tiktoken.get_encoding("cl100k_base")

ENCODING = get_encoding(MODEL)

def count_tokens(text):
    return len(ENCODING.encode(text))

def total_tokens_used(messages):
    try:
        return sum(count_tokens(msg["content"]) for msg in messages)
    except Exception as e:
        print(f"[token count error]: {e}")
        return 0

#Removes chat history over time
def enforce_token_budget(messages, budget=TOKEN_BUDGET):
    try:
        while total_tokens_used(messages) > budget:
            if len(messages) <= 2:
                break
            messages.pop(1)
    except Exception as e:
        print(f"[token budget error]: {e}")

#Response Object
def chat(user_input, temperature, max_tokens):   
    messages = st.session_state.messages
    messages.append({"role": "user", "content": user_input})
    enforce_token_budget(messages)
    
    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model = MODEL, 
            messages = messages,
            temperature = temperature,
            max_tokens = int(max_tokens)
        )
        
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply
    
#Streamlit
st.title("A Not So Friendly Chatbot")
st.sidebar.header("Options")
st.sidebar.write("This is a demo")

st.markdown(
    '<div class="app-title">😤 A Not So Friendly Chatbot</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="chip">v1 • RandomForest-powered vibe • Streamlit UI</div>',
    unsafe_allow_html=True
)
st.markdown("<hr/>", unsafe_allow_html=True)


max_tokens = st.sidebar.slider("Max Tokens", 1, 250, 100)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
system_message_type = st.sidebar.selectbox("System Message", ("Arrogant Assistant", "Custom"))

if system_message_type == "Arrogant Assistant":
    SYSTEM_PROMPT = "You are an angry and arrogant assistant who thinks humans are dumb."
elif system_message_type == "Custom":
    SYSTEM_PROMPT = st.sidebar.text_area("Custom System Message", "Customize your AI Agent here.")
else:
    SYSTEM_PROMPT = "You are a helpful assistant."
    
st.write(f"{SYSTEM_PROMPT}")

with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 🎛️ Active System Message")
    st.markdown(f"<div class='sysblock'>{SYSTEM_PROMPT}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
if st.sidebar.button("Apply New System Message"):
    st.session_state.messages[0] = {"role": "system", "content": SYSTEM_PROMPT}
    st.success("System message updated.")
    
if st.sidebar.button("Reset Conversation"):
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.success("Conversation Reset.")

if prompt := st.chat_input("What is up?"):
    reply = chat(prompt, temperature = temperature, max_tokens = max_tokens)
    
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
