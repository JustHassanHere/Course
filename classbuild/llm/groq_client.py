import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    RateLimitError,
)

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_OUTPUT_TOKENS = 4096


class GenerationError(Exception):
    """A user-presentable failure while talking to or validating the model."""


class ConfigError(GenerationError):
    """Local configuration problem (for example a missing API key)."""


@dataclass(frozen=True)
class ChatText:
    text: str
    finish_reason: str | None = None  # "length" means the token limit cut the answer


def get_model_name() -> str:
    return os.getenv("GROQ_MODEL", "").strip() or DEFAULT_MODEL


def is_reasoning_model(model: str) -> bool:
    return "gpt-oss" in model.lower()


class GroqClient:
    """Thin wrapper around the Groq SDK with translated, readable errors."""

    def __init__(self, api_key=None, model=None, timeout=REQUEST_TIMEOUT_SECONDS, sdk_client=None):
        self.model = (model or get_model_name()).strip()
        if sdk_client is not None:  # injected in tests
            self._sdk = sdk_client
            return
        api_key = (api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")).strip()
        if not api_key or api_key == "your_groq_api_key_here":
            raise ConfigError(
                "GROQ_API_KEY is missing. Add your key to the .env file "
                "(see .env.example) and restart the app."
            )
        self._sdk = Groq(api_key=api_key, timeout=timeout, max_retries=1)

    # ---- low level ---------------------------------------------------------------
    def _complete(self, **kwargs):
        try:
            return self._sdk.chat.completions.create(model=self.model, **kwargs)
        except BadRequestError:
            raise  # handled by the caller (schema fallback) or translated below
        except AuthenticationError as exc:
            raise GenerationError("Groq rejected the API key. Check GROQ_API_KEY in your .env file.") from exc
        except RateLimitError as exc:
            raise GenerationError("Groq rate limit reached. Please wait a moment and try again.") from exc
        except APITimeoutError as exc:
            raise GenerationError("The request to Groq timed out. Please try again.") from exc
        except APIConnectionError as exc:
            raise GenerationError("Could not reach Groq. Check your internet connection and try again.") from exc
        except APIStatusError as exc:
            raise GenerationError(f"Groq returned an error (status {exc.status_code}). Please try again.") from exc

    def _complete_translated(self, **kwargs):
        try:
            return self._complete(**kwargs)
        except BadRequestError as exc:
            raise GenerationError("Groq rejected the request (status 400). Please try again.") from exc

    @staticmethod
    def _read(response) -> ChatText:
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise GenerationError("The model returned an empty response. Please try again.")
        return ChatText(text=text, finish_reason=getattr(choice, "finish_reason", None))

    # ---- public -----------------------------------------------------------------
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict | None = None,
        schema_name: str = "response",
        history: list[dict] | None = None,
        temperature: float = 0.4,
        max_completion_tokens: int = MAX_OUTPUT_TOKENS,
    ) -> str:
        """Return the raw JSON text. Earlier chat turns go between system and user."""

        def messages(system: str) -> list[dict]:
            return [
                {"role": "system", "content": system},
                *(history or []),
                {"role": "user", "content": user_prompt},
            ]

        common = {"temperature": temperature, "max_completion_tokens": max_completion_tokens}
        if schema is not None:
            try:
                response = self._complete(
                    messages=messages(system_prompt),
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                    },
                    **common,
                )
                return self._read(response).text
            except BadRequestError:
                # The API refused the schema: fall back to describing it in the prompt.
                fallback = (
                    system_prompt
                    + "\n\nReturn one JSON object that matches this JSON schema exactly:\n"
                    + json.dumps(schema)
                )
                return self._read(self._complete_translated(messages=messages(fallback), **common)).text
        return self._read(self._complete_translated(messages=messages(system_prompt), **common)).text

    def chat(
        self,
        messages: list[dict],
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float = 0.4,
    ) -> ChatText:
        """Send a multi-turn conversation; the last message must be a user message."""
        if not messages:
            raise ValueError("messages must not be empty.")
        last = messages[-1]
        if last.get("role") != "user" or not str(last.get("content", "")).strip():
            raise ValueError("The last message must be a non-empty user message.")

        kwargs = {"messages": messages, "temperature": temperature}
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        if reasoning_effort and is_reasoning_model(self.model):
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        return self._read(self._complete_translated(**kwargs))


def get_client() -> GroqClient:
    return GroqClient()
