from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AZURE_AI_PROJECT_ENDPOINT: str
    AZURE_AI_PROJECT_API_KEY: str
    AZURE_AI_SERVICES_ENDPOINT: str
    SPEECH_TO_TEXT_ENDPOINT: str

    class Config:
        env_file = ".env"
        env_prefix: str = ""