from dify_plugin.entities.model import AIModelEntity, ModelType
from dify_plugin.interfaces.model import ModelProvider


class ContractModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        if credentials.get("api_key") != "valid":
            message = "invalid credentials"
            raise ValueError(message)

    def fetch_models(
        self, model_type: ModelType, credentials: dict
    ) -> list[AIModelEntity]:
        self.validate_provider_credentials(credentials)
        return [
            AIModelEntity(
                model="remote-echo", model_type=model_type, model_properties={}
            )
        ]
