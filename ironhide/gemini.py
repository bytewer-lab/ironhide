"""Gemini Agent implementation for the Ironhide framework.

This module provides the GeminiAgent class that implements chat capabilities
using Google's Gemini API, following the same interface as OpenaiAgent.
"""

import base64
import inspect
import logging
from abc import ABC
from collections.abc import Buffer, Callable
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from typing import Any, TypeVar

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, Field

from ironhide.models import Provider
from ironhide.settings import settings

logger = logging.getLogger(__name__)


class _Role(str, Enum):
    user = "user"
    model = "model"


class _FinishReason(str, Enum):
    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY = "SAFETY"
    RECITATION = "RECITATION"
    OTHER = "OTHER"


class _Modality(str, Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"


class _ResponseMimeType(str, Enum):
    application_json = "application/json"
    text_plain = "text/plain"


class _SchemaType(str, Enum):
    ARRAY = "ARRAY"
    OBJECT = "OBJECT"
    STRING = "STRING"
    INTEGER = "INTEGER"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"


class _FunctionCallingMode(str, Enum):
    AUTO = "AUTO"
    ANY = "ANY"
    NONE = "NONE"


class _FunctionCallingConfig(BaseModel):
    mode: _FunctionCallingMode
    allowed_function_names: list[str] | None = Field(
        alias="allowedFunctionNames",
        default=None,
    )


class _ToolConfig(BaseModel):
    function_calling_config: _FunctionCallingConfig = Field(
        alias="functionCallingConfig",
    )


class _PropertyDefinition(BaseModel):
    type: str
    description: str | None = None
    enum: list[str] | None = None


class _ParametersDefinition(BaseModel):
    type: str = "object"
    properties: dict[str, _PropertyDefinition]
    required: list[str] | None = None


class _FunctionDeclaration(BaseModel):
    name: str
    description: str
    parameters: _ParametersDefinition


class _Tool(BaseModel):
    function_declarations: list[_FunctionDeclaration] = Field(
        alias="functionDeclarations",
    )


class _TextPart(BaseModel):
    text: str


class _FunctionCall(BaseModel):
    name: str
    args: dict[str, Any]


class _FunctionCallPart(BaseModel):
    function_call: _FunctionCall = Field(alias="functionCall")


class _FunctionResponse(BaseModel):
    name: str
    response: dict[str, Any]


class _FunctionResponsePart(BaseModel):
    function_response: _FunctionResponse = Field(alias="functionResponse")


class _InlineData(BaseModel):
    mime_type: str = Field(alias="mimeType")
    data: str


class _ImagePart(BaseModel):
    inline_data: _InlineData = Field(alias="inlineData")


_PartType = _TextPart | _FunctionCallPart | _FunctionResponsePart | _ImagePart


class _Content(BaseModel):
    role: _Role | None = None
    parts: list[_PartType] | _PartType


class _SystemInstruction(BaseModel):
    parts: list[_TextPart]


class _SchemaDefinition(BaseModel):
    type: _SchemaType
    items: "_SchemaDefinition | None" = None
    properties: dict[str, "_SchemaDefinition"] | None = None
    property_ordering: list[str] | None = Field(alias="propertyOrdering", default=None)


class _GenerationConfig(BaseModel):
    response_mime_type: _ResponseMimeType | None = Field(
        alias="responseMimeType",
        default=None,
    )
    response_schema: _SchemaDefinition | None = Field(
        alias="responseSchema",
        default=None,
    )
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = Field(alias="maxOutputTokens", default=None)
    candidate_count: int | None = Field(alias="candidateCount", default=None)
    stop_sequences: list[str] | None = Field(alias="stopSequences", default=None)


class _TokenDetails(BaseModel):
    modality: _Modality
    token_count: int = Field(alias="tokenCount")


class _CitationSource(BaseModel):
    start_index: int = Field(alias="startIndex")
    end_index: int = Field(alias="endIndex")
    uri: str


class _CitationMetadata(BaseModel):
    citation_sources: list[_CitationSource] = Field(alias="citationSources")


class _UsageMetadata(BaseModel):
    prompt_token_count: int = Field(alias="promptTokenCount")
    candidates_token_count: int = Field(alias="candidatesTokenCount")
    total_token_count: int = Field(alias="totalTokenCount")
    prompt_tokens_details: list[_TokenDetails] | None = Field(
        alias="promptTokensDetails",
        default=None,
    )
    candidates_tokens_details: list[_TokenDetails] | None = Field(
        alias="candidatesTokensDetails",
        default=None,
    )


class _Candidate(BaseModel):
    content: _Content
    finish_reason: _FinishReason = Field(alias="finishReason")
    avg_logprobs: float | None = Field(alias="avgLogprobs", default=None)
    citation_metadata: _CitationMetadata | None = Field(
        alias="citationMetadata",
        default=None,
    )


class _GeminiRequest(BaseModel):
    contents: list[_Content]
    system_instruction: _SystemInstruction | None = None
    tools: list[_Tool] | None = None
    tool_config: _ToolConfig | None = None
    generation_config: _GenerationConfig | None = Field(
        alias="generationConfig",
        default=None,
    )


class _GeminiResponse(BaseModel):
    candidates: list[_Candidate]
    usage_metadata: _UsageMetadata = Field(alias="usageMetadata")
    model_version: str = Field(alias="modelVersion")
    response_id: str = Field(alias="responseId")


class _Headers(BaseModel):
    content_type: str = Field(alias="Content-Type", default="application/json")


class _Error(BaseModel):
    code: int
    message: str
    status: str
    details: list[dict[str, Any]] | None = None


class _ErrorResponse(BaseModel):
    error: _Error


class _Reason(BaseModel):
    thought: str


# TODO: mudar para lógica de extração de audio separada
async def audio_transcription(files: RequestFiles, api_key: str) -> str:  # noqa: ARG001
    """Transcribes audio files to text using the Gemini API.

    Args:
        files: RequestFiles object containing the audio file to transcribe.
        api_key: Gemini API key for authentication.

    Returns:
        The transcribed text as a string.

    """
    msg = "Image handling is not yet implemented for Gemini."
    raise NotImplementedError(msg)


T = TypeVar("T", bound=BaseModel)


class GeminiAgent(ABC):
    """Gemini class for implementing AI agents with chat capabilities.

    This abstract class provides the foundation for creating AI agents that can engage in chat-based interactions,
    handle structured responses, and utilize various tools and feedback mechanisms using Google's Gemini API.

    Class Variables:
        model (str): The AI model identifier to be used.
        provider (Provider): The service provider for the AI model.
        instructions (str | None): System instructions for the agent.
        chain_of_thought (tuple[str, ...] | None): Sequential prompts for thought process.
        feedback_loop (str | None): Prompt for feedback evaluation.
        messages (list[_Content]): History of chat messages.

    Methods:
        chat(input_message: str | RequestFiles, files: RequestFiles | None = None) -> str:
            Handle a chat interaction with optional audio or image processing.
        structured_chat(input_message: str | RequestFiles, response_format: type[T], files: RequestFiles | None = None) -> T:
            Handle a chat interaction with structured response validation.

    """

    model: str
    api_key: str
    instructions: str | None = None
    chain_of_thought: tuple[str, ...] | None = None
    feedback_loop: str | None = None
    messages: list[_Content]

    def __init__(
        self,
        instructions: str | None = None,
        chain_of_thought: tuple[str, ...] | None = None,
        feedback_loop: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        messages: list[_Content] | None = None,
    ) -> None:
        """Initialize the GeminiAgent with optional configuration parameters.

        Args:
            instructions: Initial system instructions for the agent.
            chain_of_thought: Sequence of thought process prompts.
            feedback_loop: Feedback evaluation prompt.
            model: AI model identifier.
            api_key: Gemini API key for authentication.
            provider: Service provider for the AI model.
            messages: Initial chat message history.

        """
        self.instructions = instructions or getattr(self, "instructions", None)
        self.chain_of_thought = chain_of_thought or getattr(
            self,
            "chain_of_thought",
            None,
        )
        self.feedback_loop = feedback_loop or getattr(self, "feedback_loop", None)
        self.model = model or getattr(self, "model", None) or settings.gemini_model
        self.api_key = (
            api_key or getattr(self, "api_key", None) or settings.gemini_api_key
        )
        self.messages = (
            messages or getattr(self, "messages", None) or self._get_history()
        )
        self.dict_tool: dict[str, Any] = {}
        self.tools = self._generate_tools()
        self.client = httpx.AsyncClient()
        self.headers = _Headers()

    def _get_history(self) -> list[_Content]:
        return []

    def _make_generation_config(
        self,
        response_format: type[BaseModel] | None,
    ) -> _GenerationConfig | None:
        def _convert_pydantic_to_gemini_schema(
            schema: dict[str, Any],
        ) -> _SchemaDefinition:
            """Convert Pydantic schema to Gemini schema format."""
            type_mapping = {
                "string": _SchemaType.STRING,
                "number": _SchemaType.NUMBER,
                "integer": _SchemaType.INTEGER,
                "boolean": _SchemaType.BOOLEAN,
                "array": _SchemaType.ARRAY,
                "object": _SchemaType.OBJECT,
            }

            schema_type = type_mapping.get(
                schema.get("type", "string"), _SchemaType.STRING
            )

            result = _SchemaDefinition(type=schema_type)

            if schema_type == _SchemaType.OBJECT and "properties" in schema:
                properties = {}
                for prop_name, prop_schema in schema["properties"].items():
                    properties[prop_name] = _convert_pydantic_to_gemini_schema(
                        prop_schema,
                    )
                result.properties = properties
                result.property_ordering = list(properties.keys())
            elif schema_type == _SchemaType.ARRAY and "items" in schema:
                result.items = _convert_pydantic_to_gemini_schema(schema["items"])

            return result

        def _remove_defaults(schema: dict[str, Any]) -> None:
            if isinstance(schema, dict):
                schema.pop("default", None)
                schema.pop("format", None)
                for value in schema.values():
                    if isinstance(value, dict):
                        _remove_defaults(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                _remove_defaults(item)

        def _remove_defs(schema: dict[str, Any]) -> None:
            if schema.get("$defs"):
                for i in schema["properties"]:
                    property_ref = schema["properties"][i].get("$ref")
                    if property_ref:
                        for j in schema["$defs"]:
                            if property_ref == f"#/$defs/{j}":
                                schema["properties"][i] = schema["$defs"][j]
                schema.pop("$defs")

        if response_format is None:
            return None

        schema = response_format.model_json_schema()
        _remove_defs(schema)
        _remove_defaults(schema)

        gemini_schema = _convert_pydantic_to_gemini_schema(schema)

        return _GenerationConfig(
            responseMimeType=_ResponseMimeType.application_json,
            responseSchema=gemini_schema,
        )

    def _generate_tools(self) -> list[_Tool]:
        tools = []
        json_type_mapping = {
            str: "string",
            int: "number",
            float: "number",
            bool: "boolean",
        }

        function_declarations = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not getattr(method, "is_tool", False):
                continue

            properties = {
                param_name: _PropertyDefinition(
                    type=json_type_mapping.get(param.annotation, "string"),
                    description=(
                        param.annotation.__metadata__[0]
                        if getattr(param.annotation, "__metadata__", None)
                        else None
                    ),
                )
                for param_name, param in inspect.signature(method).parameters.items()
                if param_name != "self"
            }

            required = [
                param_name
                for param_name, param in inspect.signature(method).parameters.items()
                if param.default is param.empty and param_name != "self"
            ]

            function_declarations.append(
                _FunctionDeclaration(
                    name=name,
                    description=(inspect.getdoc(method) or "").strip(),
                    parameters=_ParametersDefinition(
                        properties=properties,
                        required=required if required else None,
                    ),
                ),
            )
            self.dict_tool[name] = method

        if function_declarations:
            tools.append(_Tool(functionDeclarations=function_declarations))

        return tools

    async def _call_function(self, name: str, args: dict[str, Any]) -> Any:  # noqa: ANN401
        selected_tool = self.dict_tool[name]
        if inspect.iscoroutinefunction(selected_tool):
            return await selected_tool(**args)
        return selected_tool(**args)

    # TODO: isso pode ser desnecessário
    def _convert_openai_messages_to_gemini(
        self,
        messages: list[_Content],
    ) -> tuple[list[_Content], _SystemInstruction | None]:
        """Convert messages to Gemini format, extracting system instructions."""
        gemini_messages = []
        system_instruction = None

        for message in messages:
            if hasattr(message, "role") and message.role == "system":
                # Convert system messages to system instruction
                if isinstance(message.parts, list):
                    text_parts = [
                        _TextPart(text=part.text)
                        for part in message.parts
                        if hasattr(part, "text")
                    ]
                else:
                    text_parts = [_TextPart(text=str(message.parts))]
                system_instruction = _SystemInstruction(parts=text_parts)
            else:
                # Convert other messages
                gemini_messages.append(message)

        return gemini_messages, system_instruction

    async def _api_call(
        self,
        *,
        is_thought: bool = False,
        response_format: type[BaseModel] | None = None,
    ) -> _Content:
        current_response_format = _Reason if is_thought else response_format

        # Convert messages and extract system instruction
        gemini_messages, system_instruction = self._convert_openai_messages_to_gemini(
            self.messages,
        )

        # Add system instruction from self.instructions if not already set
        if not system_instruction and self.instructions:
            system_instruction = _SystemInstruction(
                parts=[_TextPart(text=self.instructions)],
            )

        data = _GeminiRequest(
            contents=gemini_messages,
            system_instruction=system_instruction,
            tools=self.tools or None,
            tool_config=(
                _ToolConfig(
                    functionCallingConfig=_FunctionCallingConfig(
                        mode=_FunctionCallingMode.NONE
                        if current_response_format
                        else _FunctionCallingMode.AUTO,
                    ),
                )
                if self.tools
                else None
            ),
            generationConfig=(
                self._make_generation_config(current_response_format)
                if current_response_format
                else None
            ),
        )

        logger.debug(
            "  >>>  Request:  %s",
            data.model_dump_json(by_alias=True, exclude_none=True, indent=4),
        )

        url = f"{settings.gemini_url.rstrip('/')}/{self.model}:generateContent"
        params = {"key": self.api_key}

        while True:
            response = await self.client.post(
                url,
                headers=self.headers.model_dump(by_alias=True),
                params=params,
                json=data.model_dump(
                    by_alias=True,
                    mode="json",
                    exclude_none=True,
                ),
                timeout=settings.ironhide_request_timeout,
            )

            if response.status_code != HTTPStatus.OK.value:
                logger.error(
                    "  >>>  Request Error:  %s",
                    data.model_dump_json(by_alias=True, exclude_none=True, indent=4),
                )
                error_response = _ErrorResponse(**response.json())
                msg = f"Gemini API error: {error_response.error.message}"
                # TODO: faltou a lógica de TOO_MANY_REQUSTS if response.status_code == HTTPStatus.TOO_MANY_REQUESTS.value:
                raise httpx.HTTPStatusError(
                    msg,
                    request=response.request,
                    response=response,
                )
            break

        gemini_response = _GeminiResponse(**response.json())
        logger.debug(
            "  >>>  Response:  %s",
            gemini_response.model_dump_json(by_alias=True, exclude_none=True, indent=4),
        )
        content = gemini_response.candidates[0].content
        self.messages.append(content)
        return content

    async def _context_provider(self, input_message: str) -> str:
        return input_message

    async def _base_chat(
        self,
        input_message: str | RequestFiles,
        response_format: type[T] | None = None,
        files: RequestFiles | None = None,
    ) -> str:
        # TODO: Adicionar lógica para diferentes serviços de extração de audio
        #### TODO: Adaptar lógica de audio da openai para gemini e grok

        processed_message: str = (
            await audio_transcription(input_message, self.api_key)
            if not isinstance(input_message, str)
            else input_message
        )

        if files:
            # TODO: Adaptar a lógica de enviar a imagem na mensagem
            await self._handle_image_message(processed_message, files)
        else:
            self.messages.append(
                _Content(
                    role=_Role.user,
                    parts=[
                        _TextPart(text=await self._context_provider(processed_message)),
                    ],
                ),
            )

        await self._handle_chain_of_thought()
        message = await self._api_call()
        message = await self._handle_tool_calls(message, response_format)
        if response_format:
            message = await self._api_call(response_format=response_format)

        response_text = ""
        if message.parts:
            if isinstance(message.parts, list):
                for part in message.parts:
                    if hasattr(part, "text"):
                        response_text += part.text
            elif hasattr(message.parts, "text"):
                response_text = message.parts.text
        return response_text

    async def _handle_image_message(
        self,
        processed_message: str,
        files: RequestFiles,
    ) -> None:
        msg = "Image handling is not yet implemented for Gemini."
        raise NotImplementedError(msg)

    async def _handle_chain_of_thought(self) -> None:
        if self.chain_of_thought:
            for thought in self.chain_of_thought:
                self.messages.append(
                    _Content(
                        role=_Role.user,
                        parts=[_TextPart(text=thought)],
                    ),
                )
                await self._api_call(is_thought=True)

    def _extract_function_calls(self, content: _Content) -> list[_FunctionCall]:
        """Extract function calls from content parts."""
        function_calls: list[_FunctionCall] = []
        if not content.parts:
            return function_calls

        parts = content.parts if isinstance(content.parts, list) else [content.parts]
        function_calls.extend(
            part.function_call for part in parts if isinstance(part, _FunctionCallPart)
        )
        return function_calls

    async def _handle_tool_calls(
        self,
        content: _Content,
        response_format: type[BaseModel] | None,
    ) -> _Content:
        # Check if content has function calls
        function_calls = self._extract_function_calls(content)

        while function_calls:
            for function_call in function_calls:
                result = await self._call_function(
                    function_call.name, function_call.args
                )
                self.messages.append(
                    _Content(
                        role=_Role.user,
                        parts=[
                            _FunctionResponsePart(
                                functionResponse=_FunctionResponse(
                                    name=function_call.name,
                                    response={"result": str(result)},
                                ),
                            ),
                        ],
                    ),
                )
            content = await self._api_call()
            function_calls = self._extract_function_calls(content)
        if response_format:
            self.messages.pop()

        return content

    async def chat(
        self,
        input_message: str | RequestFiles,
        files: RequestFiles | None = None,
    ) -> str:
        """Handle a chat interaction, optionally processing audio or image files.

        Args:
            input_message: The user's input message, which can be text or audio files.
            files: Optional image files to be included in the chat.

        Returns:
            The assistant's response as a string.

        """
        return await self._base_chat(input_message=input_message, files=files)

    async def structured_chat(
        self,
        input_message: str | RequestFiles,
        response_format: type[T],
        files: RequestFiles | None = None,
    ) -> T:
        """Handle a chat interaction with a structured response.

        Args:
            input_message: The user's input message, which can be text or audio files.
            response_format: The Pydantic model to validate and parse the response.
            files: Optional image files to be included in the chat.

        Returns:
            The assistant's response as a Pydantic model instance.

        """
        content = await self._base_chat(
            input_message=input_message,
            files=files,
            response_format=response_format,
        )
        return response_format.model_validate_json(content)


F = TypeVar("F", bound=Callable[..., Any])


def tool(func: F) -> F:
    """Mark a method as a tool that can be called by the AI model."""
    func.is_tool = True  # type: ignore[attr-defined]
    return func
