from unittest.mock import MagicMock

import pytest

from dify_plugin.core.model_factory import ModelFactory
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import AIModelEntity, ModelType
from dify_plugin.entities.model.provider import ProviderEntity, SimpleProviderEntity
from dify_plugin.interfaces.model import ModelProvider


def test_provider_dark_icons_are_preserved() -> None:
    icon_small_dark = I18nObject(en_us="small-dark.svg")
    icon_large_dark = I18nObject(en_us="large-dark.svg")
    provider = ProviderEntity(
        provider="test",
        label=I18nObject(en_us="test"),
        icon_small_dark=icon_small_dark,
        icon_large_dark=icon_large_dark,
        supported_model_types=[ModelType.LLM],
        configurate_methods=[],
    )

    simple_provider = provider.to_simple_provider()

    assert simple_provider.icon_small_dark == icon_small_dark
    assert simple_provider.icon_large_dark == icon_large_dark


def test_construct_model_provider() -> None:
    """
    Ensure ModelProvider constructor is intact and usable.
    This guards against overriding or changing __init__ signature.
    """

    class ProviderImpl(ModelProvider):
        def validate_provider_credentials(self, credentials: dict) -> None:
            pass

    provider_schema = ProviderEntity(
        provider="test",
        label=I18nObject(en_us="test"),
        supported_model_types=[ModelType.LLM],
        configurate_methods=[],
    )

    model_factory = MagicMock(spec=ModelFactory)

    provider = ProviderImpl(
        provider_schemas=provider_schema, model_factory=model_factory
    )

    assert provider is not None
    assert provider.get_provider_schema() == provider_schema
    assert provider.model_factory is model_factory
    assert provider_schema.remote_model_catalog is False
    assert provider_schema.model_dump()["remote_model_catalog"] is False
    with pytest.raises(NotImplementedError):
        provider.fetch_models(ModelType.LLM, {})

    predefined = [
        AIModelEntity(model="static", model_type=ModelType.LLM, model_properties={})
    ]
    model_factory.get_instance.return_value.predefined_models.return_value = predefined
    assert provider.models(ModelType.LLM) is predefined
    assert provider.models(ModelType.TTS) == []
    model_factory.get_instance.assert_called_once_with(ModelType.LLM)


def test_provider_remote_catalog_explicit_opt_in() -> None:
    provider_schema = ProviderEntity(
        provider="test",
        label=I18nObject(en_us="test"),
        supported_model_types=[ModelType.LLM],
        configurate_methods=[],
        remote_model_catalog=True,
    )

    assert provider_schema.remote_model_catalog is True
    assert provider_schema.model_dump()["remote_model_catalog"] is True


@pytest.mark.parametrize("remote_catalog", [False, True])
def test_simple_provider_preserves_remote_catalog_flag(remote_catalog: bool) -> None:
    provider = ProviderEntity(
        provider="test",
        label=I18nObject(en_us="test"),
        supported_model_types=[ModelType.LLM],
        configurate_methods=[],
        remote_model_catalog=remote_catalog,
    )
    simple = provider.to_simple_provider()
    assert simple.remote_model_catalog is remote_catalog
    assert simple.model_dump(mode="json")["remote_model_catalog"] is remote_catalog
    assert (
        SimpleProviderEntity.model_validate_json(
            simple.model_dump_json()
        ).remote_model_catalog
        is remote_catalog
    )


def test_simple_provider_defaults_remote_catalog_to_false() -> None:
    simple = SimpleProviderEntity(
        provider="static",
        label=I18nObject(en_us="Static"),
        supported_model_types=[ModelType.LLM],
    )
    assert simple.remote_model_catalog is False
    assert simple.model_dump()["remote_model_catalog"] is False
