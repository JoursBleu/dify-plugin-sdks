import json
import logging
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError

from dify_plugin.core.entities.plugin.request import (
    ModelActions,
    ModelFetchModelsRequest,
    ModelGetAIModelSchemas,
    PluginAccessModelRequest,
    PluginInvokeType,
)
from dify_plugin.core.model_factory import ModelFactory
from dify_plugin.core.plugin_executor import PluginExecutor
from dify_plugin.core.plugin_registration import PluginRegistration
from dify_plugin.core.runtime import Session
from dify_plugin.core.session_context import get_current_session, use_current_session
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import AIModelEntity, ModelType
from dify_plugin.entities.model.provider import ProviderEntity
from dify_plugin.errors.model import (
    InvokeAuthorizationError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model import ModelProvider
from dify_plugin.interfaces.model.ai_model import AIModel
from dify_plugin.plugin import Plugin

from .plugin_fixtures import contract_plugin, contract_source, invoke

__all__ = ["contract_plugin", "contract_source"]

PRIVATE_MARKER = "synthetic-private-catalog-marker"
SAFE_PROVIDER_ERRORS = [
    (InvokeError, "Remote model catalog provider failed"),
    (
        InvokeAuthorizationError,
        "Incorrect model credentials provided, please check and try again.",
    ),
    (InvokeConnectionError, "Connection Error"),
    (InvokeRateLimitError, "Rate Limit Error"),
    (InvokeServerUnavailableError, "Server Unavailable Error"),
]


def _model(
    model_id: str = "remote", model_type: ModelType = ModelType.LLM
) -> AIModelEntity:
    return AIModelEntity(model=model_id, model_type=model_type, model_properties={})


def _request(model_type: ModelType = ModelType.LLM) -> ModelFetchModelsRequest:
    return ModelFetchModelsRequest(
        user_id="user", provider="catalog", model_type=model_type, credentials={}
    )


def _executor(provider: ModelProvider | None) -> PluginExecutor:
    registration = MagicMock(spec=PluginRegistration)
    registration.get_model_provider_instance.return_value = provider
    return PluginExecutor(config=MagicMock(), registration=registration)


@pytest.fixture
def provider() -> MagicMock:
    instance = MagicMock(spec=ModelProvider)
    instance.get_provider_schema.return_value = ProviderEntity(
        provider="catalog",
        label=I18nObject(en_us="Catalog"),
        supported_model_types=[ModelType.LLM],
        configurate_methods=[],
        remote_model_catalog=True,
    )
    instance.fetch_models.return_value = [_model()]
    return instance


def test_request_contract_has_no_model_id() -> None:
    request = _request()
    assert ModelActions.GetAIModelList.value == "get_ai_model_list"
    assert request.type == PluginInvokeType.Model
    assert request.action == ModelActions.GetAIModelList
    assert set(ModelFetchModelsRequest.model_fields) == {
        "model_type",
        "provider",
        "credentials",
        "user_id",
        "type",
        "action",
    }
    assert not issubclass(ModelFetchModelsRequest, PluginAccessModelRequest)
    assert request.model_dump(mode="json") == {
        "type": "model",
        "action": "get_ai_model_list",
        "user_id": "user",
        "provider": "catalog",
        "model_type": "llm",
        "credentials": {},
    }
    assert "model" not in ModelFetchModelsRequest.model_json_schema()["properties"]


@pytest.mark.parametrize("field", ["user_id", "provider", "model_type", "credentials"])
def test_request_requires_provider_catalog_fields(field: str) -> None:
    data = _request().model_dump(mode="json")
    del data[field]
    with pytest.raises(ValidationError) as error:
        ModelFetchModelsRequest.model_validate(data)
    assert error.value.errors()[0]["loc"] == (field,)


def test_request_rejects_unknown_model_type() -> None:
    data = _request().model_dump(mode="json")
    data["model_type"] = "unknown"
    with pytest.raises(ValidationError):
        ModelFetchModelsRequest.model_validate(data)


@pytest.mark.parametrize("model_type", list(ModelType))
def test_catalog_is_generic_and_forwards_credentials_and_session(
    provider: MagicMock, model_type: ModelType
) -> None:
    provider.get_provider_schema.return_value.supported_model_types = [model_type]
    models = [_model("second", model_type), _model("first", model_type)]
    request = _request(model_type)
    request.credentials = {"scope": "account-a"}
    session = Session.empty_session()
    outer_session = Session.empty_session()

    def fetch(*, model_type: ModelType, credentials: dict) -> list[AIModelEntity]:
        assert model_type == request.model_type
        assert credentials is request.credentials
        assert get_current_session() is session
        return models

    provider.fetch_models.side_effect = fetch
    executor = _executor(provider)
    with use_current_session(outer_session):
        result = executor.get_ai_model_list(session, request)
        assert get_current_session() is outer_session
    assert get_current_session() is None
    assert result == {"models": models}
    assert result["models"] is models
    assert (
        TypeAdapter(dict[str, list[AIModelEntity]]).dump_python(result, mode="json")[
            "models"
        ][0]["model_type"]
        == model_type.value
    )
    provider.fetch_models.assert_called_once_with(
        model_type=model_type, credentials=request.credentials
    )
    executor.registration.get_model_provider_instance.assert_called_once_with("catalog")
    executor.registration.get_model_instance.assert_not_called()
    provider.models.assert_not_called()
    provider.validate_provider_credentials.assert_not_called()


def test_missing_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="Remote model catalog provider not found"):
        _executor(None).get_ai_model_list(Session.empty_session(), _request())


def test_catalog_requires_explicit_opt_in(provider: MagicMock) -> None:
    provider.get_provider_schema.return_value.remote_model_catalog = False
    with pytest.raises(ValueError, match="does not support remote model catalogs"):
        _executor(provider).get_ai_model_list(Session.empty_session(), _request())
    provider.fetch_models.assert_not_called()


def test_unsupported_type_does_not_fetch(provider: MagicMock) -> None:
    with pytest.raises(ValueError, match="does not support the requested model type"):
        _executor(provider).get_ai_model_list(
            Session.empty_session(), _request(ModelType.TTS)
        )
    provider.fetch_models.assert_not_called()


@pytest.mark.parametrize("size", [0, 1, 2048])
def test_catalog_accepts_empty_and_maximum_lists(
    provider: MagicMock, size: int
) -> None:
    models = [_model(str(index)) for index in range(size)]
    provider.fetch_models.return_value = models
    result = _executor(provider).get_ai_model_list(Session.empty_session(), _request())
    assert result["models"] is models


def test_catalog_rejects_overflow_before_examining_entries(provider: MagicMock) -> None:
    provider.fetch_models.return_value = [None] * 2049
    with pytest.raises(ValueError, match="at most 2048 models"):
        _executor(provider).get_ai_model_list(Session.empty_session(), _request())


@pytest.mark.parametrize("result", [None, {}, "not-a-list", (), iter(())])
def test_catalog_rejects_non_lists(provider: MagicMock, result: object) -> None:
    provider.fetch_models.return_value = result
    with pytest.raises(TypeError, match="must be a list of AIModelEntity objects"):
        _executor(provider).get_ai_model_list(Session.empty_session(), _request())


@pytest.mark.parametrize("entry", [None, {}, "invalid", 1])
def test_catalog_rejects_non_entities(provider: MagicMock, entry: object) -> None:
    provider.fetch_models.return_value = [_model(), entry]
    with pytest.raises(TypeError, match="must contain only AIModelEntity objects"):
        _executor(provider).get_ai_model_list(Session.empty_session(), _request())


def test_catalog_rejects_wrong_model_type(provider: MagicMock) -> None:
    provider.fetch_models.return_value = [_model(), _model("tts", ModelType.TTS)]
    with pytest.raises(ValueError, match="contains a different model type"):
        _executor(provider).get_ai_model_list(Session.empty_session(), _request())


def test_catalog_rejects_duplicate_ids_with_bounded_errors(provider: MagicMock) -> None:
    model_id = "private-catalog-id-" * 1024
    provider.fetch_models.return_value = [_model(model_id), _model(model_id)]
    with pytest.raises(ValueError, match="contains duplicate model IDs") as error:
        _executor(provider).get_ai_model_list(Session.empty_session(), _request())
    assert len(str(error.value)) < 100
    assert model_id not in str(error.value)


@pytest.mark.parametrize("error_type", [NotImplementedError, ValueError, RuntimeError])
def test_provider_errors_are_sanitized_and_restore_session(
    provider: MagicMock, error_type: type[Exception]
) -> None:
    failure = error_type(PRIVATE_MARKER)
    session = Session.empty_session()
    outer_session = Session.empty_session()

    def fetch(*, model_type: ModelType, credentials: dict) -> list[AIModelEntity]:
        del model_type, credentials
        assert get_current_session() is session
        raise failure

    provider.fetch_models.side_effect = fetch
    with use_current_session(outer_session):
        with pytest.raises(
            InvokeError, match=r"^Remote model catalog provider failed$"
        ) as error:
            _executor(provider).get_ai_model_list(session, _request())
        assert type(error.value) is InvokeError
        assert error.value is not failure
        assert PRIVATE_MARKER not in str(error.value)
        assert PRIVATE_MARKER not in repr(error.value.args)
        assert error.value.__suppress_context__ is True
        assert get_current_session() is outer_session
    assert get_current_session() is None


def test_opted_in_provider_without_override_returns_safe_error() -> None:
    class StaticProvider(ModelProvider):
        def validate_provider_credentials(self, credentials: dict) -> None:
            pass

    instance = StaticProvider(
        provider_schemas=ProviderEntity(
            provider="catalog",
            label=I18nObject(en_us="Catalog"),
            supported_model_types=[ModelType.LLM],
            configurate_methods=[],
            remote_model_catalog=True,
        ),
        model_factory=MagicMock(spec=ModelFactory),
    )
    with pytest.raises(InvokeError, match=r"^Remote model catalog provider failed$"):
        _executor(instance).get_ai_model_list(Session.empty_session(), _request())
    assert get_current_session() is None


def _wire_request(**overrides: object) -> dict:
    return {
        **_request().model_dump(mode="json"),
        "provider": "contract_models",
        "credentials": {"api_key": "valid"},
        **overrides,
    }


def test_registered_catalog_route_serializes_models_without_model_id(
    contract_plugin: Plugin,
) -> None:
    request = _wire_request()
    assert "model" not in request
    assert sum(route.filter(request) for route in contract_plugin.routes) == 1
    assert not any(
        route.filter({**request, "type": "tool"}) for route in contract_plugin.routes
    )
    provider = contract_plugin.registration.get_model_provider_instance(
        "contract_models"
    )
    assert provider.get_provider_schema().remote_model_catalog is True
    messages = invoke(contract_plugin, request)
    assert len(messages) == 1
    assert messages[0]["type"] == "stream"
    assert set(messages[0]["data"]) == {"models"}
    assert messages[0]["data"]["models"][0]["model"] == "remote-echo"
    assert messages[0]["data"]["models"][0]["model_type"] == "llm"
    assert provider.models(ModelType.LLM)[0].model == "echo"


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"provider": "missing"}, "ValueError", "Invalid remote model catalog request"),
        (
            {"model_type": "text2img"},
            "ValueError",
            "Invalid remote model catalog request",
        ),
        ({"credentials": {}}, "InvokeError", "Remote model catalog provider failed"),
        (
            {"model_type": "unknown"},
            "ValidationError",
            "Invalid remote model catalog request",
        ),
    ],
)
def test_registered_catalog_errors_end_without_partial_models(
    contract_plugin: Plugin, overrides: dict, error_type: str, message: str
) -> None:
    messages = invoke(contract_plugin, _wire_request(**overrides))
    assert len(messages) == 1
    assert messages[0]["type"] == "error"
    assert messages[0]["data"]["error_type"] == error_type
    assert message in messages[0]["data"]["message"]
    assert "models" not in messages[0]["data"]


@pytest.mark.parametrize(("error_type", "message"), SAFE_PROVIDER_ERRORS)
def test_provider_errors_keep_only_canonical_categories(
    provider: MagicMock, error_type: type[InvokeError], message: str
) -> None:
    class PrivateProviderError(error_type):
        def __str__(self) -> str:
            pytest.fail("Untrusted provider exception must never be stringified")
            return super().__str__()

    failure = PrivateProviderError(PRIVATE_MARKER)
    provider.fetch_models.side_effect = failure
    with pytest.raises(error_type) as error:
        _executor(provider).get_ai_model_list(Session.empty_session(), _request())
    assert type(error.value) is error_type
    assert str(error.value) == message
    assert error.value.description == message
    assert error.value.args == (message,)
    assert error.value.__suppress_context__ is True


@pytest.mark.parametrize("stage", ["registration", "provider_schema"])
def test_provider_setup_errors_are_sanitized(provider: MagicMock, stage: str) -> None:
    executor = _executor(provider)
    target = (
        executor.registration.get_model_provider_instance
        if stage == "registration"
        else provider.get_provider_schema
    )
    target.side_effect = RuntimeError(PRIVATE_MARKER)
    with pytest.raises(
        InvokeError, match=r"^Remote model catalog provider failed$"
    ) as error:
        executor.get_ai_model_list(Session.empty_session(), _request())
    assert PRIVATE_MARKER not in repr(error.value.args)
    provider.fetch_models.assert_not_called()


@pytest.mark.parametrize(("error_type", "message"), SAFE_PROVIDER_ERRORS)
def test_remote_schema_provider_errors_are_sanitized_without_changing_static_schema(
    provider: MagicMock, error_type: type[InvokeError], message: str
) -> None:
    executor = _executor(provider)
    model = MagicMock(spec=AIModel)
    failure = error_type(PRIVATE_MARKER)
    model.get_model_schema.side_effect = failure
    executor.registration.get_model_instance.return_value = model
    request = ModelGetAIModelSchemas(
        user_id="user",
        provider="catalog",
        model="remote",
        model_type=ModelType.LLM,
        credentials={"api_key": PRIVATE_MARKER},
    )
    with pytest.raises(error_type) as error:
        executor.get_ai_model_schemas(Session.empty_session(), request)
    assert type(error.value) is error_type
    assert str(error.value) == message
    assert PRIVATE_MARKER not in repr(error.value.args)
    provider.get_provider_schema.return_value.remote_model_catalog = False
    with pytest.raises(error_type) as static_error:
        executor.get_ai_model_schemas(Session.empty_session(), request)
    assert static_error.value is failure


def _assert_private_error_absent(
    messages: list[dict], caplog: pytest.LogCaptureFixture
) -> None:
    assert PRIVATE_MARKER not in json.dumps(messages)
    assert "traceback" not in json.dumps(messages)
    assert PRIVATE_MARKER not in caplog.text
    for record in caplog.records:
        assert record.exc_info is None
        assert record.stack_info is None
        assert PRIVATE_MARKER not in repr(record.__dict__)


@pytest.mark.parametrize("action", ["get_ai_model_list", "get_ai_model_schemas"])
@pytest.mark.parametrize(("error_type", "message"), SAFE_PROVIDER_ERRORS)
def test_registered_catalog_provider_errors_never_log_or_serialize_private_data(
    contract_plugin: Plugin,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    action: str,
    error_type: type[InvokeError],
    message: str,
) -> None:
    provider = contract_plugin.registration.get_model_provider_instance(
        "contract_models"
    )
    failure = error_type(PRIVATE_MARKER)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    request = _wire_request(action=action, credentials={"api_key": PRIVATE_MARKER})
    if action == "get_ai_model_list":
        monkeypatch.setattr(provider, "fetch_models", fail)
    else:
        model = contract_plugin.registration.get_model_instance(
            "contract_models", ModelType.LLM
        )
        monkeypatch.setattr(type(model), "get_model_schema", fail)
        request["model"] = "remote"
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="dify_plugin")
    messages = invoke(contract_plugin, request)
    assert messages == [
        {
            "type": "error",
            "data": {
                "error_type": error_type.__name__,
                "message": message,
                "args": {"description": message},
            },
        }
    ]
    _assert_private_error_absent(messages, caplog)


@pytest.mark.parametrize("action", ["get_ai_model_list", "get_ai_model_schemas"])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, NotImplementedError])
def test_registered_untrusted_provider_errors_are_bounded_and_private(
    contract_plugin: Plugin,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    action: str,
    error_type: type[Exception],
) -> None:
    provider = contract_plugin.registration.get_model_provider_instance(
        "contract_models"
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise error_type(PRIVATE_MARKER * 1024)

    request = _wire_request(action=action, credentials={"api_key": PRIVATE_MARKER})
    if action == "get_ai_model_list":
        monkeypatch.setattr(provider, "fetch_models", fail)
    else:
        model = contract_plugin.registration.get_model_instance(
            "contract_models", ModelType.LLM
        )
        monkeypatch.setattr(type(model), "get_model_schema", fail)
        request["model"] = "remote"
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="dify_plugin")
    messages = invoke(contract_plugin, request)
    assert messages == [
        {
            "type": "error",
            "data": {
                "error_type": "InvokeError",
                "message": "Remote model catalog provider failed",
                "args": {"description": "Remote model catalog provider failed"},
            },
        }
    ]
    assert len(json.dumps(messages)) < 250
    assert get_current_session() is None
    _assert_private_error_absent(messages, caplog)


@pytest.mark.parametrize("field", ["model_type", "provider", "credentials", "user_id"])
def test_registered_malformed_catalog_request_is_sanitized_before_executor(
    contract_plugin: Plugin, caplog: pytest.LogCaptureFixture, field: str
) -> None:
    request = _wire_request(credentials={"api_key": PRIVATE_MARKER})
    request[field] = (
        PRIVATE_MARKER
        if field in {"model_type", "credentials"}
        else {"private": PRIVATE_MARKER}
    )
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="dify_plugin")
    messages = invoke(contract_plugin, request)
    assert messages == [
        {
            "type": "error",
            "data": {
                "error_type": "ValidationError",
                "message": "Invalid remote model catalog request",
                "args": {},
            },
        }
    ]
    _assert_private_error_absent(messages, caplog)


@pytest.mark.parametrize("instance_route", [False, True])
@pytest.mark.parametrize("default_writer", [False, True])
def test_router_catalog_type_error_reaches_sanitized_boundary_without_early_output(
    contract_plugin: Plugin,
    caplog: pytest.LogCaptureFixture,
    instance_route: bool,
    default_writer: bool,
) -> None:
    class InvalidRequest:
        def __init__(self, **_data: object) -> None:
            raise TypeError(PRIVATE_MARKER)

    def handler(session: Session, data: InvalidRequest) -> None:
        del session, data
        pytest.fail("Malformed catalog request must not execute")

    def instance_handler(
        _instance: object, session: Session, data: InvalidRequest
    ) -> None:
        handler(session, data)

    contract_plugin.routes = []
    contract_plugin.response_writer = MagicMock() if default_writer else None
    contract_plugin.register_route(
        instance_handler if instance_route else handler,
        lambda data: data.get("action") == "get_ai_model_list",
        instance=object() if instance_route else None,
    )
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="dify_plugin")
    messages = invoke(
        contract_plugin, _wire_request(credentials={"api_key": PRIVATE_MARKER})
    )
    assert messages == [
        {
            "type": "error",
            "data": {
                "error_type": "TypeError",
                "message": "Invalid remote model catalog request",
                "args": {},
            },
        }
    ]
    if default_writer:
        contract_plugin.response_writer.error.assert_not_called()
    _assert_private_error_absent(messages, caplog)


def test_registered_catalog_passes_app_session_and_credentials(
    contract_plugin: Plugin, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = contract_plugin.registration.get_model_provider_instance(
        "contract_models"
    )
    seen: list[dict] = []

    def fetch(*, model_type: ModelType, credentials: dict) -> list[AIModelEntity]:
        session = get_current_session()
        assert session is not None
        assert session.session_id == "session-1"
        assert session.app_id == "app-catalog"
        assert model_type == ModelType.LLM
        seen.append(credentials)
        return [_model(credentials["scope"])]

    monkeypatch.setattr(provider, "fetch_models", fetch)
    for scope in ("account-a", "account-b"):
        messages = invoke(
            contract_plugin,
            _wire_request(credentials={"scope": scope}),
            app_id="app-catalog",
        )
        assert messages[0]["data"]["models"][0]["model"] == scope
    assert seen == [{"scope": "account-a"}, {"scope": "account-b"}]
    assert get_current_session() is None


@pytest.mark.parametrize("invalid_catalog", ["duplicate", "overflow", "entity", "type"])
def test_registered_catalog_validation_reports_one_error(
    contract_plugin: Plugin, monkeypatch: pytest.MonkeyPatch, invalid_catalog: str
) -> None:
    provider = contract_plugin.registration.get_model_provider_instance(
        "contract_models"
    )
    results = {
        "duplicate": [_model("same"), _model("same")],
        "overflow": [None] * 2049,
        "entity": [_model(), {}],
        "type": [_model("tts", ModelType.TTS)],
    }
    monkeypatch.setattr(provider, "fetch_models", lambda **_: results[invalid_catalog])
    messages = invoke(contract_plugin, _wire_request())
    assert len(messages) == 1
    assert messages[0]["type"] == "error"
    assert messages[0]["data"]["error_type"] in {"ValueError", "TypeError"}
    assert "models" not in messages[0]["data"]
