from ..mcts import MCTS
import logwood
import numpy as np

class EvaluationResults:
    """Stores result information from evaluations"""

    def __init__(self, incumbent_mcts, challenger_mcts, win_threshold=0.51):
        self.incumbent = incumbent_mcts
        self.challenger = challenger_mcts
        self.threshold = win_threshold

        self.incumbent_wins = 0
        self.challenger_wins = 0
        self.draws = 0


    def get_winning_mcts(self):
        if self._challenger_won():
            return self.challenger
        else:
            return self.incumbent


    @property
    def winner(self):
        if self._challenger_won:
            return "Challenger"
        else:
            return "Incumbent"
            

    def _challenger_won(self):
        return ((
            self.challenger_wins \
            / (self.incumbent_wins + self.challenger_wins)) \
             > self.threshold)

    


class NNEvaluator:

    def __init__(self, environment, config, win_threshold=0.55):
        self._logger = logwood.get_logger(self.__class__.__name__)
        self.environment = environment.clone()
        self.config = config
        self.win_threshold = 0.55

    def _validate_config(self, config):
        """Validates that the configuraton file is valid.
        
        Arguments:
            config {dictionary} -- A policy configuration dictionary
        
        Raises:
            ValueError -- If validation fails
        """
        if config.get('terminal'):
            raise ValueError("Evaluators should not have terminal policies.")



    def evaluate(self, incumbent_model, challenger_model, games=100):
        """Evaluates a tournament between MCTS guided by 
        two neural networks.
        
        Arguments:
            incumbent {mcts.nn.incumbent.BaseNN} -- The current incumbent
            challenger_model {mcts.nn.incumbent.BaseNN} -- The challenger incumbent
        
        Keyword Arguments:
            games {int} -- Number of games to play in tournament (default: {100})
        
        Returns:
            EvaluationResults -- A results class detailing the game results
        """
        # TODO: Update config so we can pass the calculation time in there
        incumbent = MCTS(self.environment, name='Evaluation Incumbent', calculation_time=2)
        challenger = MCTS(self.environment, name='Evaluation Challenger', calculation_time=2)

        self._logger.debug("Building MCTS")
        incumbent.build(self.config)
        challenger.build(self.config)

        self._logger.debug("Setting Models")
        incumbent.set_policy_attribute(("model", incumbent_model))
        challenger.set_policy_attribute(("model", challenger_model))

        self._logger.debug("Initializing Results")
        results = EvaluationResults(incumbent, challenger, win_threshold=0.55)

        for game in range(games):
            self._logger.debug("Starting game: {}".format(game))
            # Alternate who starts first.
            while not self.environment.terminal:
                if game % 2:
                    incumbent.act()
                    if self.environment.terminal: continue
                    challenger.act()
                else:
                    challenger.act()
                    if self.environment.terminal: continue
                    incumbent.act()
            
            winner = self.environment.winner
            if winner == None:
                results.draws += 1
            elif (winner == 0 and game % 2 == 0) or \
                (winner == 1 and game % 2 != 0):
                results.incumbent_wins += 1
            else:
                results.challenger_wins += 1
            self.environment.reset()
        
        return results









    