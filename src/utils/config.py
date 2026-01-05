from omegaconf import DictConfig, OmegaConf
from skyrl_train.utils import validate_cfg as validate_cfg_skyrl

from src.tools import DEFAULT_OPENHANDS_TOOLS, TOOL_REGISTRY

def register_resolvers():
    OmegaConf.register_new_resolver("replace", lambda s,old,new: s.replace(old, new))
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))

def validate_cfg(config: DictConfig):
    validate_cfg_skyrl(config)

    assert hasattr(
        config, "generator"
    ), "Missing `generator` config block"

    assert hasattr(
        config.generator, "tools"
    ), "Missing `generator.tools` (pick via Hydra `agent=...`)"

    assert (
        config.generator.tools is not None and len(config.generator.tools) > 0
    ), "`generator.tools` must be a non-empty list"

    for tool in config.generator.tools:
        assert tool in DEFAULT_OPENHANDS_TOOLS or tool in TOOL_REGISTRY, f"Tool {tool} does not exist in the registry"

    assert hasattr(
        config.generator, "prompts"
    ), "Missing `generator.prompts` (pick via Hydra `prompt=...`)"

    assert hasattr(
        config.generator.prompts, "system_prompt"
    ), "Missing `generator.prompts.system_prompt`"

    assert hasattr(
        config.generator.prompts, "user_prompt"
    ), "Missing `generator.prompts.user_prompt`"

