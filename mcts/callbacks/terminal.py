from ..base.policy import NodeTrackingPolicy
import numpy as np

class StagedModelTrainer(NodeTrackingPolicy):
    """Operates in three stages.
    
    1. Data Generation
    While the MCTS plays games, data is added to the replay table. This happens until
    the terminal policy has been called a set number of times.
    2. Training
    Once n_games have been reached, begins training. This will occupy the main thread
    until training is finished. The length of this phase is defined by n_training_steps.
    3. Evaluation
    The evaluation phase """
    def __init__(self, model, replay, trainer, evaluator,
                generation_steps=1000, training_steps=300, evaluation_steps=400):
        
        super().__init__()

        self.generation_model = model
        self.training_model = model.clone()

        self.replay = replay
        self.trainer = trainer
        self.evaluator = evaluator

        self.trainer.set_model(self.training_model)
    
        self.generation_steps = generation_steps
        self._current_step = 0
        self.training_steps = training_steps
        self.evaluation_steps = evaluation_steps

    def __call__(self, game_history, reward, winner):
        self._logger.info("Adding data to replay table")
        for node in game_history:
            node = self.tree.get_by_id(node)
            state = node.state

            # The action values have three possible sources of information.
            # 1. The edge for that action value was traversed.
            #    In this case, we simply use the action-value on the node-edge.
            # 2. The edge was not traversed but is a valid action.
            #    In this case, we want a zero-gradient in the learning process.
            #    To accomplish that, we just set the action value to what our
            #    model would expect.
            # 3. The edge is not a valid action. In this case we set the action-value
            #    To the -1 (absolute loss) to discourage the model from preferring these
            #    actions.
            priors, _ = self.training_model.predict_from_node(node)
            
            action_values = priors[0]
            valid_actions = node.edges
            for i in range(len(priors)):
                if i not in valid_actions:
                    action_values[i] = -1
                elif node[i].n != 0:
                    action_values[i] = node[i].q


            if node.player != winner:
                reward = -reward
            
            self.replay.add_data(state, action_values, reward)  

        if self._current_step > self.generation_steps:
            self._logger.info("Entering the training phase.")
            self.train()

            self._logger.info("Entering the evaluatoin phase.")
            self.evaluate()

        self._current_step += 1

    def train(self):
        """Start the training phase"""
        self.trainer.train_batches(self.training_steps)

    def evaluate(self):
        """Start the evaluation phase"""
        # Build the MCTS
        results = self.evaluator.evaluate( 
                    self.generation_model, 
                    self.training_model,
                    games=self.evaluation_steps)

        # Update the generation model if it won
        if results.winner == 'challenger':
            self._logger.info("Training Model Wins! Updating generation model.")
            self.generation_model.set_weights(self.training_model.get_weights())
        