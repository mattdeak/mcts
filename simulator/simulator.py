class Simulator:

    def __init__(self, env):
        self.environment = env

    def simulate(self, players, n=10):
        env = self.environment # Shortcut
        players = {i: players[i] for i in range(len(players))}
        sim_data = {i: {'wins': 0, 'losses': 0} for i in range(len(players))}

        for i in range(n):
            env.reset()
            while not env.terminal:
                current_player = players[env]
                winner = current_player.act()
            sim_data[winner]['wins'] += 1

            for player in sim_data:
                if player != winner:
                    sim_data[player]['losses'] += 1

        return sim_data
