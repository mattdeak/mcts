import logwood
from logwood.handlers.stderr import ColoredStderrHandler

logwood.basic_config(
        level = logwood.INFO,
        handlers = [ColoredStderrHandler()]
)