import os
from jinja2 import Environment, FileSystemLoader

path = os.path.dirname(__file__)

def get_instruction(
    instance: dict,
    prompt_path: str,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
    workspace_dir_name = instance["repo"].split("/")[-1]

    # Set up Jinja2 environment
    if prompt_path is None:
        prompt_path = os.path.join(path, "templates", "default.j2")
    prompts_dir = os.path.dirname(prompt_path)
    template_name = os.path.basename(prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
        "working_dir": workspace_path,
    }
    # context["test_instructions"] = ""

    # Render the instruction
    instruction = template.render(context)
    return instruction