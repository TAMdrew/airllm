from transformers import GenerationConfig

from .airllm_base import AirLLMBaseModel


class AirLLMGemma(AirLLMBaseModel):

    def __init__(self, *args, **kwargs):
        super(AirLLMGemma, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()

class AirLLMGemma2(AirLLMBaseModel):

    def __init__(self, *args, **kwargs):
        super(AirLLMGemma2, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()

class AirLLMGemma3(AirLLMBaseModel):

    def __init__(self, *args, **kwargs):
        super(AirLLMGemma3, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()

class AirLLMGemma4(AirLLMBaseModel):

    def __init__(self, *args, **kwargs):
        super(AirLLMGemma4, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()
