import logwood
from logwood.handlers.stderr import ColoredStderrHandler

# Configure logging
logwood.basic_config(
        level = logwood.INFO,
        handlers = [ColoredStderrHandler()]
)

SUPPORTED_POLICY_TYPES = {
    'action' : ['most-visited','proportional-to-visit-count'],
    'selection': ['ucb1', 'puct'],
    'expansion' : ['vanilla','neural'],
    'simulation' : ['to-end'],
    'update' : ['vanilla', 'value'],
    'expansion_rollout': ['random-unvisited','random']
}
