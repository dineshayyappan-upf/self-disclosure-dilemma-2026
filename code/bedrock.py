"""Recorded Bedrock request adapter and visible-output extraction."""
import json
import re
import time

ANTHROPIC_VERSION = "bedrock-2023-05-31"


def strip_reasoning_tags(text):
    """Strip <reasoning>...</reasoning> and <think>...</think> blocks from model output.
    GPT-OSS and Qwen models wrap chain-of-thought in these tags — we want only
    the final answer for embedding comparison."""
    cleaned = re.sub(r'<(?:reasoning|think)>.*?</(?:reasoning|think)>', '', text,
                     flags=re.DOTALL)
    return cleaned.strip()


def bedrock_invoke(client, model_id, messages, system=None,
                   max_tokens=1024, temperature=0.7, retries=3,
                   provider="anthropic"):
    """
    Invoke a model via Bedrock with retry logic.
    Supports Anthropic (Claude) and OpenAI-compatible (GPT-OSS, Qwen) formats.
    """
    if provider == "anthropic":
        body = {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system
    else:
        # OpenAI-compatible format (GPT-OSS, Qwen, etc.)
        # GPT-OSS and Qwen use reasoning tokens that count against max_tokens,
        # so we need a much higher limit (reasoning can easily eat 500-1000 tokens)
        effective_max = max(max_tokens, 4096)
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)
        body = {
            "messages": oai_messages,
            "max_tokens": effective_max,
            "temperature": temperature,
        }

    for attempt in range(retries):
        try:
            response = client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())

            if provider == "anthropic":
                text = result["content"][0]["text"]
                usage = result.get("usage", {})
                return text, {
                    "input": usage.get("input_tokens", 0),
                    "output": usage.get("output_tokens", 0),
                }
            else:
                # OpenAI format: choices[0].message.content
                text = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                return text, {
                    "input": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                    "output": usage.get("completion_tokens", usage.get("output_tokens", 0)),
                }

        except Exception as e:
            if "ThrottlingException" in str(type(e).__name__) or "throttl" in str(e).lower():
                wait = 2 ** attempt * 5
                print(f"    [THROTTLED] Waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            elif attempt < retries - 1:
                print(f"    [ERROR] {e} — retrying ({attempt+1}/{retries})...")
                time.sleep(2)
            else:
                raise
