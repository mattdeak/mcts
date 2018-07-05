from .policies.action import *
from .policies.expansion import *
from .policies.selection import *
from .policies.update import *
from .policies.rollout import *
from .policies.simulation import *
import inspect


class ConfigBuilder:

    _CLASS_LOOKUP = {
        'action' : {
            'most-visited' : MostVisited,
            'proportional-to-visit-count' : ProportionalToVisitCount
        },
        'selection' : {
            'ucb1' : UCB1,
            'puct' : PUCT
        },
        'expansion' : {
            'vanilla' : VanillaExpansion,
            'neural' : NNExpansion
        },
        'simulation' : {
            'random-to-end' : RandomToEnd
        },
        'update' : {
            'vanilla' : VanillaUpdate,
            'value' : ValueUpdate
        },
        'expansion_rollout' : {
            'random' : RandomChoice,
            'random-unvisited' : RandomUnvisited
        },
    }

    @classmethod
    def build(cls, config):
        built_configuration = {}
        model = config.get('model')

        for key in cls._CLASS_LOOKUP.keys():
            kwargs = config.get(key + '_kwargs')
            config_choice = config.get(key)

            if config_choice:
                policy = cls._CLASS_LOOKUP[key][config_choice]
                needs_model = cls._policy_needs_model(policy)

                # Instantiate the policy with relevant arguments
                if kwargs:
                    if needs_model:
                        built_policy = policy(model, **kwargs)
                    else:
                        built_policy = policy(**kwargs)
                else:
                    if needs_model:
                        built_policy = policy(model)
                    else:
                        built_policy = policy()
                built_configuration[key] = built_policy

        return built_configuration

    @staticmethod
    def _policy_needs_model(c):
        args = inspect.getargspec(c)
        return 'model' in args.args


            
