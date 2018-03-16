class Simulator:

    def __init__(self, env):
        # Make sure we have a deep copy
        self.environment = env.clone()

    def simulate(self, players, n=10):
        env = self.environment # Shortcut
        for player in players:
            player.environment=env

        player_dict = {i+1: players[i] for i in range(len(players))}
        sim_data = {i+1: {'wins': 0, 'losses': 0, 'draws':0} for i in range(len(players))}

        for i in range(n):
            print(f"Starting game {i}")
            print(env.board)
            for player in player_dict.values():
                if player.terminal:
                    player.reset()
            while not env.terminal:
                current_player = player_dict[env.player]
                print(f"Player {env.player} Turn")
                current_player.act()


            winner = env.winner
            if winner:
                print(f"Winner player {winner}")
                sim_data[winner]['wins'] += 1

                for player in sim_data:
                    if player != winner:
                        sim_data[player]['losses'] += 1
            else:
                for player in sim_data:
                    sim_data[player]['draws'] += 1

        return sim_data
