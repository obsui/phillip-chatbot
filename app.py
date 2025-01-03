import gradio as gr
import requests
from huggingface_hub import InferenceClient
import os

# Fetch the Hugging Face API token from environment variables
HF_API_TOKEN = os.getenv("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")
if not HF_API_TOKEN:
    raise ValueError("Hugging Face API token not found. Please set HF_API_TOKEN as an environment variable.")

# Initialize the HuggingFace Inference Client
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_API_TOKEN)

# Fetch the CoinMarketCap API key from environment variables
CMC_API_KEY = os.getenv("df92670d-41c3-4740-81a6-9477d10e24bd")
if not CMC_API_KEY:
    raise ValueError("CoinMarketCap API key not found. Please set CMC_API_KEY as an environment variable.")

CMC_BASE_URL_LATEST = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
HEADERS = {"X-CMC_PRO_API_KEY": "df92670d-41c3-4740-81a6-9477d10e24bd"}

def get_top_movers_1h():
    """
    Fetch the top movers based on 1-hour percent change from CMC API.
    """
    params = {
        "start": "1",
        "limit": "10",  # Fetch the top 10 coins
        "sort": "percent_change_1h",  # Sort by 1-hour percent change
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
        return coins if coins else ["No top movers found! Sorry brother."]
    except requests.RequestException as e:
        return [f"Error: {str(e)}"]

def respond(message, history, system_message, max_tokens, temperature, top_p):
    """
    Debug-friendly version of the respond function.
    """
    try:
        # Add system context
        messages = [{"role": "system", "content": system_message}]

        # Add previous conversation history
        for entry in history:
            if "user" in entry and "assistant" in entry:
                messages.append({"role": "user", "content": entry["user"]})
                messages.append({"role": "assistant", "content": entry["assistant"]})
            else:
                return [{"role": "assistant", "content": "Error: Invalid history format."}]

        # Add user message
        messages.append({"role": "user", "content": message})

        # Check for specific commands
        if any(keyword in message.lower() for keyword in ["coin", "market", "ticker", "pump", "memecoin"]):
            coins = get_top_movers_1h()
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "\n".join(coins)})
            return history

        # Generate response via Hugging Face API
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

        # Add response to history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response.strip()})
        return history

    except Exception as e:
        return [{"role": "assistant", "content": f"Error: {str(e)}"}]

# Gradio Chat Interface setup with type="messages"
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(
            value="You are Phillip, a crypto advisor and expert. You use the words 'bro', 'bruh', 'man', and 'brother' frequently in your responses, along with other Gen Z-Internet and Crypto slang. Always keep responses concise, casual, and engaging.",
            label="System message"
        ),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)"
        ),
    ],
    title="Crypto & Chill Bot",
    description="Your crypto buddy is here to chat, share insights, and keep things chill. Ask about top movers or anything else!",
    type="messages"  # This ensures the messages are treated correctly
)

# Launch the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to port 8080 if none is provided
    demo.launch(server_port=port, server_name="0.0.0.0")