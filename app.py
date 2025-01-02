#!/usr/bin/env python3

import gradio as gr
import requests
from huggingface_hub import InferenceClient
import os

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

CMC_API_KEY = "your-cmc-api-key"
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
        return coins if coins else ["No top movers found."]
    except requests.RequestException as e:
        return [{"role": "assistant", "content": f"Error: {str(e)}"}]

def respond(message, history, system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_message}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    if any(keyword in message.lower() for keyword in ["coin", "market", "ticker", "pump", "memecoin"]):
        coins = get_top_movers_1h()
        return [{"role": "assistant", "content": f"Sure thing. Here's what's moving rn!:\n" + "\n".join(coins)}]

    if message.lower() in ["yerr", "yo", "sup", "hey"]:
        return [{"role": "assistant", "content": "Sup? Lmk if you need anything crypto or just wanna chat!"}]

    messages.append({"role": "user", "content": message})
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

    return [{"role": "assistant", "content": response.strip()}]

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(
            value="You are Phillip, a crypto advisor. You respond using 'bro', 'bruh' and 'man' frequently. You communicate using Gen-Z and crypto internet slang. Keep responses concise, casual, and engaging.",
            label="System message"
        ),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p"),
    ],
    title="Phillip - Your degen crypto bot",
    description="Phillip keeps diamond hands while providing insights.",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_port=port, server_name="0.0.0.0")