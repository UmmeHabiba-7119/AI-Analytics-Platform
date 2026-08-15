import requests

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

MODEL_NAME = "google/gemma-4-e4b"


def call_llm(messages, temperature=0.2, max_tokens=700):
    """
    Send messages to the locally running Gemma model
    through the LM Studio OpenAI-compatible API.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(
            LM_STUDIO_URL,
            json=payload,
            timeout=600
        )

        response.raise_for_status()

        result = response.json()

        message = result["choices"][0]["message"]

        content = message.get("content", "").strip()

        finish_reason = result["choices"][0].get(
            "finish_reason"
        )

        if not content:
            if finish_reason == "length":
                return (
                    "The model output was truncated. "
                    "Please try again with a shorter question."
                )

            return "The model returned an empty response."

        return content

    except requests.exceptions.ConnectionError:
        return (
            "Could not connect to LM Studio. "
            "Please make sure the local server is running."
        )

    except requests.exceptions.Timeout:
        return "The model took too long to respond."

    except requests.exceptions.RequestException as error:
        return f"LM Studio API error: {error}"

    except Exception as error:
        return f"Unexpected error: {error}"

def clean_generated_code(text: str):
    """
    Remove Markdown code fences from LLM-generated code.
    """

    cleaned = text.strip()

    if cleaned.startswith("```python"):
        cleaned = cleaned[len("```python"):]

    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()