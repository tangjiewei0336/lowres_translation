#!/usr/bin/env python3
from openai import OpenAI


def main() -> None:
    base_url = "http://localhost:8005/v1"
    model = "Qwen/Qwen3.5-9B"

    client = OpenAI(
        api_key="EMPTY",
        base_url=base_url,
    )

    text = "原神启动"

    messages = [
        {
            "role": "system",
            "content": "You are a professional translator. Translate Simplified Chinese to English.",
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=12800,
            temperature=0.0,
            extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    },

        )
        print(resp)
        out = resp.choices[0].message.content
        print("源句：", text)
        print("译文：", out)
    except Exception as e:
        print("调用失败：", e)


if __name__ == "__main__":
    main()

