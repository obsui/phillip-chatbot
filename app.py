import gradio as gr
import requests
from huggingface_hub import InferenceClient
import os

# Fetch API tokens from environment variables
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("Hugging Face API token not found. Please set HF_API_TOKEN environment variable")
CMC_API_KEY = os.getenv("CMC_API_KEY")
if not CMC_API_KEY:
    raise ValueError("CoinMarketCap API key not found. Please set CMC_API_KEY environment variable")

# Initialize the HuggingFace Inference Client
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_API_TOKEN)
CMC_BASE_URL_LATEST = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
HEADERS = {"X-CMC_PRO_API_KEY": CMC_API_KEY}

def get_top_movers_1h():
    params = {
        "start": "1",
        "limit": "10",
        "sort": "percent_change_1h",
        "convert": "USD",
    }
    try:
        response = requests.get(CMC_BASE_URL_LATEST, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        coins = [
            f"{coin['name']} ({coin['symbol']}): ${coin['quote']['USD']['price']:.2f}, 1h Change: {coin['quote']['USD']['percent_change_1h']:.2f}%"
            for coin in data["data"]
        ]
        return "\n".join(coins) if coins else "No top movers found! Sorry brother."
    except requests.RequestException as e:
        return f"Error: {str(e)}"

def respond(message, history, system_message, max_tokens, temperature, top_p):
    try:
        # Create a more specific system message that prevents self-dialogue
        enhanced_system_message = system_message + "\nIMPORTANT: You must only respond as the assistant. Never generate or include user messages in your responses. Wait for the user to ask questions and respond directly to them. Do not create fictional dialogue or responses from the user."
        
        # Add system context
        messages = [{"role": "system", "content": enhanced_system_message}]
        
        # Add conversation history with explicit roles
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Check for crypto-related keywords
        if any(keyword in message.lower() for keyword in ["coin", "market", "ticker", "pump", "memecoin"]):
            return get_top_movers_1h()
        
        # Generate response via Hugging Face API with role enforcement
        response = ""
        for msg in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = msg.choices[0].delta.content
            response += token
            
        # Clean up any potential user-style messages that might slip through
        cleaned_response = response.strip()
        if "<|user|>" in cleaned_response or "User:" in cleaned_response:
            cleaned_response = cleaned_response.split("<|user|>")[0].strip()
            cleaned_response = cleaned_response.split("User:")[0].strip()
            
        return cleaned_response
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Chat Interface setup
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(
            value="You are Phillip, a crypto advisor and expert. You use the words 'bro', 'bruh', 'man', and 'brother' in your responses, along with other Gen Z-Internet and Crypto slang primarily taken from Twitter and Tumblr. Always keep responses concise, casual, and engaging. Only respond to what the user says - never generate user messages or create dialogue.",
            label="System message"
        ),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p")
    ],
    title="",
    description="Phil is here to chat, share insights, and hang out. Ask about coins, the market, tickers, or anything else!"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_port=port, server_name="0.0.0.0")
