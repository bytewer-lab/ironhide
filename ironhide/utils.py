from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, create_model


def get_prompt(file_path: str, prompt: str) -> str:
    """Read and return the content of a prompt file.

    Args:
        file_path: Path to the main file.
        prompt: Name of the prompt file (relative to the main file's directory).

    Returns:
        The content of the prompt file.

    """
    system_file_path = Path(file_path).parent / prompt
    with system_file_path.open(encoding="utf-8") as system_file:
        return system_file.read()


class Provider(str, Enum):
    """Enumeration of supported AI service providers."""

    openai = "openai"
    gemini = "gemini"
    grok = "grok"
    anthropic = "anthropic"
    deepseek = "deepseek"
    qwen = "qwen"


PROVIDER_URLS: dict[Provider, str] = {
    Provider.openai: "https://api.openai.com/v1/",
    Provider.gemini: "https://generativelanguage.googleapis.com/v1beta/",
    Provider.grok: "https://api.x.ai/v1/",
    Provider.anthropic: "https://api.anthropic.com/v1/",
    Provider.deepseek: "https://api.deepseek.com/",
    Provider.qwen: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/",
}


def create_class_from_schema(schema: dict[str, Any]) -> type[BaseModel]:
    """Recria classes Pydantic a partir do schema JSON gerado por model_json_schema().

    Args:
        schema: O dicionário retornado pelo método model_json_schema()

    Returns:
        A classe Pydantic principal reconstruída

    """
    # Armazenar as classes definidas para referência
    class_registry = {}

    # Processa as definições de classes aninhadas ($defs)
    if "$defs" in schema:
        for class_name, class_schema in schema["$defs"].items():
            class_registry[class_name] = _create_single_class(
                class_schema,
                class_name,
                class_registry,
            )

    # Cria a classe principal
    return _create_single_class(
        schema,
        schema.get("title", "MainModel"),
        class_registry,
    )


def _create_single_class(
    class_schema: dict[str, Any],
    class_name: str,
    class_registry: dict[str, type],
) -> type[BaseModel]:
    """Cria uma única classe Pydantic a partir do schema.

    Args:
        class_schema: O schema para esta classe específica
        class_name: O nome da classe
        class_registry: Registro de classes já criadas

    Returns:
        Nova classe Pydantic

    """
    # Preparar os campos para a classe
    fields = {}

    if "properties" in class_schema:
        for prop_name, prop_schema in class_schema["properties"].items():
            # Verifica se é uma referência a outra classe
            if "$ref" in prop_schema:
                # Extrai o nome da classe referenciada
                ref_name = prop_schema["$ref"].split("/")[-1]
                field_type = class_registry[ref_name]
                description = prop_schema.get("description", "")
                fields[prop_name] = (field_type, Field(description=description))
            else:
                # Determina o tipo Python apropriado
                python_type = _get_python_type(prop_schema)
                description = prop_schema.get("description", "")
                fields[prop_name] = (python_type, Field(description=description))

    # Cria a classe usando create_model
    new_class: type[BaseModel] = create_model(class_name, **fields)
    # Adiciona a classe ao registro
    class_registry[class_name] = new_class

    return new_class


def _get_python_type(prop_schema: dict[str, Any]) -> type:
    """Determina o tipo Python apropriado com base no schema da propriedade.

    Args:
        prop_schema: O schema para esta propriedade

    Returns:
        O tipo Python correspondente

    """
    schema_type = prop_schema.get("type")
    schema_format = prop_schema.get("format")

    if schema_type == "string":
        if schema_format == "date":
            return date
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        items_type = _get_python_type(prop_schema.get("items", {"type": "string"}))
        return list[items_type]
    if schema_type == "object":
        return dict[str, Any]
    # Fallback para qualquer outro tipo
    return Any
