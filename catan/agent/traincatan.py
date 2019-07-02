import os
import pickle
import datetime

import numpy as np
from matplotlib import pyplot as plt

from game.game import Game
from agent.rl import DeepQNetwork

class TrainCatan:
    """
    A class describing a Training Session of a Catan Game.

    ...

    Attributes
    ----------
    plot_interval : int
        plotting resolution in number of games
    show_cards_statistic : bool
        whether the statistics are plotted
    action_space : str
        action space defined in Game class
    output_graph : bool
        whether tensorflow data shall be gathered locally
    position_training_instances : tuple(0/1)
    	1 if the player in this position shall participate as training instance
    random_shuffle_training_players_ : bool
        whether different players shall participate in training
    needed_victory_points : int
        amount of victory points to finish the game (see Game class)
    reward : str
        reward used during trainint (see Game class)
    learning_rate : float
        initial learning rate
    learning_rate_decay_factor : float
        exponential decay factor of the learning rate
    learning_rate_start_decay : int
        amount of games after which the learning rate starts decaying
    reward_decay : float
        reward decay factor (gamma, see DeepQNetwork class)
    e_greedy : float
        maximum epsilon (see DeepQNetwork class, currently not used)
    replace_target_iter : int
        amount of network updates performed before replacing the target network by the eval network
    memory_size : int
        amount of state transitions (s,a,r,s') which will maximally be stored in RAM
    num_games : int
        amount of games played in the training session
    final_epsilon : float
        maximum epsilon (see DeepQNetwork class, currently not used)
    epsilon_increase : int
        amount of games after which epsilon starts increasing (currently not used)
    softmax_choice : bool
        whether the taken action shall be chosen according to the softmax values of the q values
    list_num_neurons : tuple(int,int...)
        dqn architecture definition, each integer creates a layer with the corresponding amount of neurons (see DeepQNetwork class)
    batch_size : int
        batch size used for weight updates (see DeepQNetwork class)
    training_players : np.array(int,...)
        list of the player numbers particiopating in training
    state_space_buffer : list(states)
        list of the last visited state by each player participating in training
    action_buffer : list(int)
        list of the last taken action index by each player participating in training
    reward_buffer : list(float)
        list of the last obtained reward by each player participating in training
    sigmoid_001_099_borders : tuple(int,int)
        amount of games played so that the sigmoidal (epsilon) function crosses 0.01 / 0.99 respectively
    autosave : bool
        whether the dqn model shall be saved automatically when the average win reate reaches a new maximum
    random_init : bool
        whether the board in initialized randomly (see Game class)
    verbose : bool
        whether text output shall be generated to visualize some parts of the training statistics
    print_episodes : bool
        whether the current amount of played games shall be printed to a file (needed for Distributed Instances Training)
    victories : list(int)
        list of player numbers who won a game
    one_of_training_instances_wins : list(binary)
        1 if the game was won by a training player, 0 otherwise
    cards : list(int)
        list of player numbers with most cards in the corresponding game
    epsilons : list(float)
        list of epsilon values for each game number
    learning_rates : list(float)
        list of learning rate values for each game number
        
    
    Methods
    -------
    save_hyperparameters(filename)
        Counts the player number up by one and rolls the dices
    load_hyperparameters(filename)
        Loads the hyperparameters of the TrainCatan instance 'hyperparameters\filename'.
    init_taken_action_storage()
        Initializes lists which are used for counting the actions each player took during training.
    add_action_to_storage(clabel,player)
        Adds an action to a players action storage
    print_stored_actions()
        Prints the actions stored for each player
    random_shuffle_training_players()
        Returns an array with a random number between 0 and 3
    start_training(training = True)
        Starts the current training session.
    gather_statistics(env,iteration_counter,training,episode)
        Prints and stores the statistics after plot_interval has been reached
        
    """
    def __init__(self,plot_interval=100,action_space='buildings_only',position_training_instances = (1,0,0,0),
                 needed_victory_points = 3,reward = 'victory_only',
                 learning_rate=1,
                 learning_rate_decay_factor = 0.999,
                 learning_rate_start_decay = 4000,
                 reward_decay=0.95,
                 e_greedy=0,
                 replace_target_iter=20,
                 memory_size=50000,
                 num_games=10000,
                 final_epsilon=0.9,
                 epsilon_increase=1000, #since which game the epsilon shall start to increase exponentially
                 softmax_choice= False,
                 sigmoid_001_099_borders = (-1000,5000), #epsilon sigmoid function
                 opponents = 'random_sample',
                 autosave = True,
                 random_shuffle_training_players = False, # Shall the training player positions be randomized?
                 random_init = False,# Shall the game board be randomly initialized?
                 show_cards_statistic = False,
                 list_num_neurons = (50,50),
                 verbose = True,
                 print_episodes = False,
                 batch_size = 256,
                 output_graph = False,
                 activation_function = 'relu',
                 loss_function = 'mse',
                 optimizer_function = 'gradient'
                 ):
        self.plot_interval = plot_interval
        self.show_cards_statistic = show_cards_statistic
        self.action_space = action_space
        self.output_graph = output_graph
        self.position_training_instances = position_training_instances
        self.random_shuffle_training_players_ = random_shuffle_training_players
        self.needed_victory_points,self.reward = needed_victory_points,reward
        self.learning_rate,self.learning_rate_decay_factor,self.learning_rate_start_decay=learning_rate,learning_rate_decay_factor,learning_rate_start_decay
        self.reward_decay,self.e_greedy,self.replace_target_iter,self.memory_size = reward_decay,e_greedy,replace_target_iter,memory_size
        self.num_games = num_games
        self.final_epsilon = final_epsilon
        self.epsilon_increase = epsilon_increase
        self.softmax_choice = softmax_choice
        self.list_num_neurons = list_num_neurons
        self.batch_size = batch_size

        self.init_training_environment(activation_function,loss_function,optimizer_function)
        self.training_players = np.where(np.array(position_training_instances)==1)[0]
        self.state_space_buffer=[None,None,None,None]
        self.action_buffer=[None,None,None,None]
        self.reward_buffer=[None,None,None,None]

        self.sigmoid_001_099_borders = sigmoid_001_099_borders
        self.autosave = autosave
        
        self.random_init = random_init
        self.victories = []
        self.one_of_training_instances_wins = []
        self.cards = []
        self.epsilons = []
        self.learning_rates = []

        self.verbose = verbose
        self.print_episodes = print_episodes



    def save_hyperparameters(self,filename):
        """
        Saves the hyperparameters of this TrainCatan instance.

        The RL model is NOT saved but saved separately.
        The class is written to file with all members.

        :param filename:
        	filename under which the hyperparameters are saved
        """
        del(self.RL)
        if not os.path.exists('hyperparameters'):
            os.makedirs('hyperparameters')
        if not os.path.exists('hyperparameters/'+filename):
            os.makedirs('hyperparameters/'+filename)
        f = open('hyperparameters/'+filename+'/'+filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()


    def load_hyperparameters(self,filename):
        """
        Loads the hyperparameters of the TrainCatan instance 'hyperparameters\filename'.

        The RL model is NOT loaded.
        All members of the class are loaded from the file.

        :param filename:
        	filename from which to load the hyperparameters.
        :returns:
            TrainCatan Instance
        """
        if not os.path.exists('hyperparameters/'+filename):
            return
        f = open('hyperparameters/'+filename+'/'+filename, 'rb')
        return pickle.load(f)

    def init_taken_action_storage(self):
        """
        Initializes lists which are used for counting the actions each player took during training.

        """
        self.donothing = [0,0,0,0]
        self.trade4vs1 = [0,0,0,0]
        self.buildroad = [0,0,0,0]
        self.buildsettlement = [0,0,0,0]
        self.buildcity = [0,0,0,0]
        self.trade3vs1 = [0,0,0,0]
        self.trade2vs1 = [0,0,0,0]

    def add_action_to_storage(self,clabel,player):
        """
        Adds an action to a players action storage

        :param clabel:
        	label of the action taken
        :param player:
            player number

        """
        if clabel == 'build_road':
            self.buildroad[player]+=1
        elif clabel == 'build_settlement':
            self.buildsettlement[player]+=1
        elif clabel == 'build_city':
            self.buildcity[player]+=1
        elif clabel == 'trade_4vs1':
            self.trade4vs1[player]+=1
        elif clabel == 'trade_3vs1':
            self.trade3vs1[player]+=1
        elif clabel == 'trade_2vs1':
            self.trade2vs1[player]+=1
        elif clabel == 'do_nothing':
            self.donothing[player]+=1

    def print_stored_actions(self):
        """
        Prints the actions stored for each player

        """
        print('Do Nothing: '+str(self.donothing))
        print('Trade4vs1: '+str(self.trade4vs1))
        print('Trade3vs1: '+str(self.trade3vs1))
        print('Trade2vs1: '+str(self.trade2vs1))
        print('BuildRoad: '+str(self.buildroad))
        print('BuildSettlement: '+str(self.buildsettlement))
        print('BuildCity: '+str(self.buildcity))
    
    def random_shuffle_training_players(self):
        """
        Returns an array with a random number between 0 and 3

        """
        # Returns the binary representation of the training players
        # DID NOT WORK TOO WELL rand_num = np.random.randint(15)+1
        # DID NOT WORK TOO WELL return np.where(np.unpackbits(np.array(rand_num, dtype=np.uint8))[-4:]==1)[0]
        # Rather train only one player per round?
        rand_num = np.random.randint(4)
        return np.array([rand_num])
         
    
    def start_training(self,training = True):
        """
        Starts the current training session.

        Initializes the plot and records and plots all data gathered during
        training. Manages the learning rate and epsilon increases.
        Manages the interface to the DQN model by storing transitions taken in the games
        and by obtaining the taken actions from the model.

        :param training:
        	Indicates whether a model shall be trained. If false the current RL model
            will be used playing games with an epsilon value of 1.
        """
        self.init_taken_action_storage()
        self.init_online_plot()
        self.init_epsilon_function()
        step = 0
        for episode in range(self.num_games):
            # initial observation, get state space

            self.env = Game(random_init=self.random_init,action_space=self.action_space,needed_victory_points=self.needed_victory_points,reward=self.reward)
            env = Game(random_init=self.random_init,action_space=self.action_space,needed_victory_points=self.needed_victory_points,reward=self.reward)
            
            state_space = env.get_state_space()
            possible_actions = env.get_possible_actions(env.current_player)
            
            iteration_counter = 0
            self.state_space_buffer=[None,None,None,None]
            self.action_buffer=[None,None,None,None]
            self.reward_buffer=[0,0,0,0]
            self.done_buffer=[None,None,None,None]
            
            while True:
                # fresh env
                # env.render()

                iteration_counter += 1
                # RL choose action based on state

                if env.current_player-1 in self.training_players:
                    buffer_player = env.current_player-1

                    self.action_buffer[buffer_player] = self.RL.choose_action(state_space,possible_actions)
                    state_space_, self.reward_buffer[buffer_player], possible_actions, self.done_buffer[buffer_player],clabel = env.step(self.action_buffer[buffer_player])
                    if np.all(np.array(self.done_buffer)[self.training_players]==1):
                        self.RL.store_transition(state_space, self.action_buffer[buffer_player], self.reward_buffer[buffer_player], state_space_)
                    if env.current_player-1 != buffer_player: #When player one chooses do Nothing
                        self.state_space_buffer[buffer_player] = state_space
                    else:
                        if training:
                            self.RL.store_transition(state_space, self.action_buffer[buffer_player], self.reward_buffer[buffer_player], state_space_)
                else:
                    action = np.random.choice(len(possible_actions), 1, p=possible_actions/sum(possible_actions))[0]
                    state_space_, r, possible_actions, d ,clabel= env.step(action)

                    if env.current_player-1 in self.training_players:
                        buffer_player = env.current_player-1
                        if self.state_space_buffer[buffer_player] is not None and self.action_buffer[buffer_player] is not None and training:
                            self.RL.store_transition(self.state_space_buffer[buffer_player], self.action_buffer[buffer_player], self.reward_buffer[buffer_player], state_space_)

                self.add_action_to_storage(clabel,env.current_player-1)

                # The game executes the action chosen by RL and gets next state and reward

                if (step > 2000) and (step % 50 == 0) and training:
                    self.RL.learn()

                # swap observation
                state_space = state_space_

                # break while loop when end of this episode
                step += 1
                
                if np.all(np.array(self.done_buffer)[self.training_players]==1):
                    self.gather_statistics(env,iteration_counter,training,episode)
                    if (len(self.victories)%self.plot_interval==0) and (episode>0):

                        if self.random_shuffle_training_players_:
                            self.training_players=self.random_shuffle_training_players()
                        
                        self.plot_statistics_online(self.victories, self.epsilons,self.cards,self.one_of_training_instances_wins,self.learning_rates,self.plot_interval)
                        if self.print_episodes:
                            with open('episodes', 'w') as f:
                                f.write(str(episode))
                                f.close()
                    break

        plt.show()
        # end of game
        print('Run Finished')
        self.print_stored_actions()

    def gather_statistics(self, env,iteration_counter,training,episode):
        """
        Prints and stores the statistics after plot_interval has been reached

        :param env:
        	current Game instance.
        :param iteration_counter:
            current move number taken by all players across game numbers
        :param training:
            whether the DQN model shall be trained
        :param episode:
            current game number
        """
        if self.verbose:
            print(self.reward_buffer)
            print('Game '+ str(episode)+' finished after ' + str(iteration_counter)+' iterations.####################################################')
            print('Victory Points ' +str(env.get_victory_points()))
            print('Cards '+str(np.sum(env.cards,axis=1)))
        if training :
            self.RL.epsilon = np.tanh((episode-self.eps_mid)*self.eps_stretch_factor)*0.5+0.5
        self.RL.lr = self.learning_rate_decay(episode)
        if self.verbose:
            print('Epsilon '+str(self.RL.epsilon)+'\n')
        self.cards.append(np.argmax(np.sum(env.cards,axis=1)))
        self.victories.append(np.argmax(env.get_victory_points()))
        #self.one_of_training_instances_wins.append(np.sum(np.array(self.reward_buffer))/len(self.training_players))
        if self.reward == 'cards':
            self.one_of_training_instances_wins.append(np.argmax(np.sum(env.cards,axis=1))==self.training_players[0])
        else:
            self.one_of_training_instances_wins.append(np.where(self.victories[-1]==self.training_players[0],1,0))
        self.epsilons.append(self.RL.epsilon)
        self.learning_rates.append(self.RL.lr)

    def play_game(self,position_training_instances = (1,0,0,0),epsilon=1.0):
        if not hasattr(self,'RL'):
            print('No reinforcement learning instance available.')
            return
        self.RL.epsilon = epsilon
        self.training_players = np.where(np.array(position_training_instances)==1)[0]
        print(self.training_players)
        self.start_training(training=False)


    def init_training_environment(self,activation_function,loss_function,optimizer_function):
        env = Game(action_space=self.action_space)
        self.RL = DeepQNetwork(len(env.get_possible_actions(1)), len(env.get_state_space()), # total action, total features/states
                      learning_rate=self.learning_rate,
                      reward_decay=self.reward_decay,
                      e_greedy=self.e_greedy,
                      replace_target_iter=self.replace_target_iter,
                      memory_size=self.memory_size,
                      softmax_choice=self.softmax_choice,
                      batch_size=self.batch_size,
                      list_num_neurons = self.list_num_neurons,
                      output_graph = self.output_graph,
                      activation_function= activation_function,
                      loss_function=loss_function,
                      optimizer_function=optimizer_function
                      )

    def init_online_plot(self,title='Figure',plot_counter = 0,make_new_figure = True):
        if make_new_figure:
            plt.figure(plot_counter)
            plt.title(title)
        plt.plot([],[],label='Player 1 vic')
        plt.plot([],[],label='Player 2 vic')
        plt.plot([],[],label='Player 3 vic')
        plt.plot([],[],label='Player 4 vic')
        if self.show_cards_statistic:
            plt.plot([],[],label='Player 1 cards')
            plt.plot([],[],label='Player 2 cards')
            plt.plot([],[],label='Player 3 cards')
            plt.plot([],[],label='Player 4 cards')
        plt.plot([],[],label='Epsilon')
        plt.plot([],[],label='Average Win Rate')
        plt.plot([],[],label='Learning Rate')
        plt.legend()
        plt.xlabel('Game number')
        plt.ylabel('Winning percentage / Epsilon value')


    def calc_averaged_statistics(self,victories,epsilons,cards,win_rate,learning_rate,n_game_average):
        start_ind = 0
        end_ind = 0
        avg_vic = []
        avg_cards = []
        avg_eps = []
        avg_win_rate = []
        avg_lr = []
        num_games = []
        while True:
            end_ind += n_game_average
            if end_ind > len(victories):
                end_ind = len(victories)
            num_games.append(end_ind-n_game_average/2)
            vic_extract = np.array(victories[start_ind:end_ind])
            cards_extract = np.array(cards[start_ind:end_ind])
            eps_extract = epsilons[start_ind:end_ind]
            win_rate_extract = win_rate[start_ind:end_ind]
            lr_extract = learning_rate[start_ind:end_ind]

            avg_vic.append([sum(np.where(vic_extract==0,1,0))/len(vic_extract),sum(np.where(vic_extract==1,1,0))/len(vic_extract)
                               ,sum(np.where(vic_extract==2,1,0))/len(vic_extract),sum(np.where(vic_extract==3,1,0))/len(vic_extract)])
            avg_cards.append([sum(np.where(cards_extract==0,1,0))/len(cards_extract),sum(np.where(cards_extract==1,1,0))/len(cards_extract)
                                 ,sum(np.where(cards_extract==2,1,0))/len(cards_extract),sum(np.where(cards_extract==3,1,0))/len(cards_extract)])
            avg_eps.append(np.mean(eps_extract))

            avg_win_rate.append(np.mean(win_rate_extract))
            avg_lr.append(np.mean(lr_extract))
            if end_ind == len(victories):
                break
            start_ind = end_ind
        avg_vic = np.array(avg_vic)
        avg_cards = np.array(avg_cards)

        return avg_vic,avg_cards,avg_lr,avg_eps,avg_win_rate,num_games

    def plot_statistics_online(self,victories,epsilons,cards,win_rate,learning_rate,n_game_average):
        avg_vic,avg_cards,avg_lr,avg_eps,avg_win_rate,num_games = self.calc_averaged_statistics(victories,epsilons,cards,win_rate,learning_rate,n_game_average)
        
        if self.autosave:
            if avg_win_rate[-1] >= np.max(avg_win_rate):
                self.RL.save_current_model(str(datetime.date.today()))

        for i in range(4):
            plt.gca().lines[i].set_xdata(num_games)
            plt.gca().lines[i].set_ydata(avg_vic[:,i])
        if self.show_cards_statistic:
            for i in range(4):
                plt.gca().lines[i+4].set_xdata(num_games)
                plt.gca().lines[i+4].set_ydata(avg_cards[:,i])
        plt.gca().lines[4+self.show_cards_statistic*4].set_xdata(num_games)
        plt.gca().lines[4+self.show_cards_statistic*4].set_ydata(avg_eps)
        plt.gca().lines[5+self.show_cards_statistic*4].set_xdata(num_games)
        plt.gca().lines[5+self.show_cards_statistic*4].set_ydata(avg_win_rate)
        plt.gca().lines[6+self.show_cards_statistic*4].set_xdata(num_games)
        plt.gca().lines[6+self.show_cards_statistic*4].set_ydata(avg_lr)
        plt.gca().relim()
        plt.gca().autoscale_view()
        plt.pause(0.05)

    def learning_rate_decay(self,episode):
        if episode < self.learning_rate_start_decay:
            return self.learning_rate
        self.learning_rate*=self.learning_rate_decay_factor
        return self.learning_rate

    def init_epsilon_function(self):
        self.eps_mid = np.mean(self.sigmoid_001_099_borders)
        self.eps_stretch_factor = np.arctanh((0.99-0.5)/0.5)/(self.sigmoid_001_099_borders[1]-self.eps_mid)

