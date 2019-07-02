
# used to shuffle the deck of hexes
import random
import itertools
import numpy as np

from game.roads import Roads
from game.crossings import Crossings
from game.hex_tiles import HexTiles
from game import defines



class Game:
	"""
    A class describing a Catan game

    ...

    Attributes
    ----------
    needed_victory_points : int
        amount of victory points which have to be reached to finish the game
    reward : str
        string describing the type of reward which will be returned when performing a game step
    tiles : list(tuple(int,int))
        description of the tiles of the game board, HexTiles instance
    crossings : list(tuple(...))
        list with one tuple for each crossing, where the first tuple entry is a list of all crossing indices connected
        to that crossing, the second is the list of the tiles connected to that crossing and the third the information
    	which harbour is connected to that crossing, Crossings instance
    roads : list(tuple(int,int))
    	Roads instance
    building_state : list(int)
    	list with an integer describing if and which kind of building is placed on each crossing
    dices_results : list(int)
    	numbers which can be rolled by two combined 6 sided dices
    cards : np.array([4,5])
    	each row contains the current amount of cards of each of the five resources for each of the four players
    current_player : int
    	the current player number between 1 and 4
    robber : int
    	index of the tile where the robber is currently placed
    seven_rolled : int
    	0 - seven is not rolled in the current turn, 1 - seven is rolled in the current turn
    rob_player_state : int
    	0 - indicates that a player is not robbed in the next turn
    	1 - indicates that a player is robbed in the next turn
    dev_cards : np.array([4,5])
    	each row contains the current amount of development cards of each of the five types of development cards held
    	in hand by each of the four players
    dev_cards_discovered : np.array([4,5])
    	each row contains the current amount of development cards of each of the five types of development cards
    	already played by a player and visible for all other players
    dev_cards_playable : np.array([4,5])
    	each row contains the current amount of development cards of each of the five types of development cards which
    	are playable by each of the players
    dev_cards_stack : list(int)
    	representing the stack of development cards
    action_counter : int
    	the amount of actions taken during the course of the game
    action_array_names_dic : dict(str:list(fun,fun))
		dictionary containing a string describing the subset of the whole action set as key - and a list containing
		a function for obtaining the subset of the possible action in the current state and the corresponding function
		for executing the chosen action.
	state_array_dic : dict(str:fun)
		dictionary containing a string describing a subset of the whole state space as key - and a function
		which returns this subspace as value

    Methods
    -------
    next_players_turn()
        Counts the player number up by one and rolls the dices
    get_possible_action_do_nothing()
        Returns a 1 if skipping the turn is allowed and a 0 otherwise (e.g. when you HAVE to move the robber)
    step(action)
        Make a game step by taking the action
    get_possible_actions(player_num)
        Concatenates and return all possible actions of the given player.
    take_action(chosen_action_ind,player_num)
        Executes the action chosen by the RL algorithm.
    count_up_action_counter()
        Counts up the games taken action counter and controls the intial placement order.
    get_state_space()
        Returns the current state space
    roll_dice()
        Rolls the dices.
    move_robber(action_ind,player_num)
        Moves the robber to the tile with number action_ind
    distribute_resources(number)
        Distribute the resources according to the rules.
    add_resource(resource_type,player_num)
        Adds a resource of resource_type to the stack of cards of player player_num.
    check_resource_available_on_pile(resource_type)
        Checks if a certain resource type is still available to be drawn by a player.
    place_settlement(crossing_index,player_num)
        Places a settlement at the crossing index position.
    distribute_resources_second_settlement(crossing_index,player_num)
        Distributes resources of second settlement according to the rules.
    place_road(road_index,player_num)
        Places a road on the given road index.
    place_city(crossing_index,player_num)
        Places a city on the given crossing index.
    rob_player(self,robbed_player_index,player_num)
        Takes a random card from player robbed_player_index and passes it to player player_num.
    get_possible_actions_build_settlement(player_num)
        Returns all locations where settlement can be placed by the given player.
    get_possible_actions_build_road(self,player_num)
        Returns a vector of 0's and 1's with 1's on indices where roads can be placed.
    check_resources_available(player_num,buying_good)
        Check availability of resources for the specific buying good.
    pay(player_num,buying_good)
        Reduces the cards of others player by the amount needed for the resources.
    get_possible_actions_build_city(player_num)
        Returns all locations where cities can be placed by the given player.
    set_robber_position(tile_number)
        Puts the robber on the specified tile.
    get_possible_action_move_robber(player_num):
        Returns all locations where the robber can be placed.
    get_robber_state()
        Returns the location of the robber in state representation.
    rob_person(player_num)
        Return list of players possible to rob resource from
    get_possible_actions_rob_player(player_num)
        Return list of players possible to rob resource from.
    get_possible_actions_trade_bank(player_num)
        Return list of possible trades to be done with the bank.
    get_possible_actions_trade_3vs1(player_num)
        Return list of possible trades to be done via 3vs1 trade.
    create_possible_trade_sets_3vs1()
        Creates all possible sets of 3 vs 1 tradings.
    has_3vs1_port(player_num)
        Get information on whether the player has a settlement/city adjacent to a 3vs1 port.
    has_2vs1_port(player_num)
        Get information on whether the player has a settlement/city adjacent to a 2vs1 port.
    get_possible_actions_trade_2vs1(player_num)
        Return list of possible trades to be done with a port.
    discard_resources()
        Simple heuristic for discarding cards.
    create_possible_actions_dictionary()
        According to initialization of the game, only certain action spaces will be added to this dictionary.
    trade_bank(action_index,player_num)
        Executes the action trade with bank for a certain player and a certain action index.
    trade_3vs1(action_index,player_num)
        Executes the action trade 3vs1 for a certain player and a certain action index
    trade_2vs1(action_index,player_num)
        Executes the action trade 2vs1 for a certain player and a certain action index
    get_victory_points()
        Calculates the current victory points for each player
    """


	def __init__(self,action_space='buildings_only',random_init=True,needed_victory_points=8,reward='victory_only'):
		self.needed_victory_points = needed_victory_points
		self.reward = reward
		self.tiles = HexTiles(random_init)
		self.crossings = Crossings(self.tiles.get_tiles(),self.tiles.harbours)
		self.roads = Roads(self.crossings.get_neighbouring_crossings())
		self.crossings.create_connected_roads(self.roads.get_roads())
		self.building_state = self.crossings.get_building_state()
		self.dices_results = [2,3,3,4,4,4,5,5,5,5,6,6,6,6,6,7,7,7,7,7,7,8,8,8,8,8,9,9,9,9,10,10,10,11,11,12]

		# 4 players, 5 resources
		self.cards = np.zeros((4,5))

		# Sets the first players turn
		self.current_player = 1

		# Initializes the robber
		self.robber = self.tiles.get_desert_hex()
		self.seven_rolled = 0
		self.rob_player_state = 0

		# Initialize dev_card stack

		self.dev_cards = np.zeros((4, 5))
		self.dev_cards_discovered = np.zeros((4, 5))
		self.dev_cards_playable = np.zeros((4, 5))

		self.dev_card_stack = [defines.DEV_KNIGHT] * 14 + [defines.DEV_VICTORYPOINT] * 5 \
							+ [defines.DEV_MONOPOLY] * 2 + [defines.DEV_YEAROFPLENTY] * 2 \
							+ [defines.DEV_ROADBUILDING] * 2
		random.shuffle(self.dev_card_stack)

		# Action Counter
		self.action_counter = 20

		# List of possible action arrays
		# Key: action space name, Value: [get_possible_actions_function, execute_corresponding_action_function]
		self.action_array_names_dic = {'do_nothing':[self.get_possible_action_do_nothing,self.next_players_turn],
									   'build_road':[self.get_possible_actions_build_road,self.place_road],
									   'build_settlement':[self.get_possible_actions_build_settlement,self.place_settlement],
									   'build_city':[self.get_possible_actions_build_city,self.place_city]}
		assert type(action_space) is str
		if action_space == 'buildings_only':
			# Trading necessary, otherwise game can get stuck as not all resources would be available
			self.action_array_names_dic['trade_4vs1']=[self.get_possible_actions_trade_bank,self.trade_bank]
			pass
		elif action_space == 'building_and_trade':
			self.action_array_names_dic['trade_4vs1']=[self.get_possible_actions_trade_bank,self.trade_bank]
			self.action_array_names_dic['trade_3vs1']=[self.get_possible_actions_trade_3vs1,self.trade_3vs1]
			self.action_array_names_dic['trade_2vs1']=[self.get_possible_actions_trade_2vs1,self.trade_2vs1]
		elif action_space == 'building_trade_and_rob':
			self.action_array_names_dic['trade_4vs1']=[self.get_possible_actions_trade_bank,self.trade_bank]
			self.action_array_names_dic['trade_3vs1']=[self.get_possible_actions_trade_3vs1,self.trade_3vs1]
			self.action_array_names_dic['trade_2vs1']=[self.get_possible_actions_trade_2vs1,self.trade_2vs1]
			self.action_array_names_dic['move_robber']=[self.get_possible_action_move_robber,self.move_robber]
			self.action_array_names_dic['rob_player']=[self.get_possible_actions_rob_player,self.rob_player]




		self.state_array_dic = {'tile_state':self.get_state_space_tiles, #number space included in tile_state as in Data Representation
								#'number_state':self.get_state_space_numbers,
								'port_state':self.get_state_space_ports,
								'building_state':self.get_state_space_buildings,
								'road_state':self.get_state_space_roads,
								'robber_state':self.get_state_space_robber,
								'cards_state':self.get_state_space_cards}

		self.create_possible_actions_dictionary()
		self.create_possible_trade_sets_3vs1()
		self.create_number_space()
		self.create_tile_space()
		self.create_port_space()

		# Action Counter starts at 0. This will make sure that the 4 players will go through the intialization phase
		# as all actions apart of build_settlement and build_road are disabled.
		self.action_counter = 0


	def next_players_turn(self,action_ind,player_num):
		"""
        Counts the player number up by one and rolls the dices

        :param player_num:
        	not used
        :param action_ind:
        	not used
        """
		self.current_player = self.current_player + 1 if self.current_player < 4 else 1
		self.roll_dice()

	def get_possible_action_do_nothing(self,player_num):
		"""
        Return the action array do_nothing

        A one if skipping the turn is allowed and a zero otherwise (e.g. when you HAVE to move the robber)

        :returns:
        	numpy array with possible actions, length 1 (1's and 0's).
        """
		if self.seven_rolled or self.rob_player_state or self.action_counter < 16:
			return np.array([0])
		else:
			return np.array([1])


	def step(self,action):
		"""
        Make a game step by taking the action.

        Returns the resulting state space, the reward, a game_finished flag and the 
        label describing the action which was taken.
        
        :param action:
        	action index
        :returns:
        	resulting state space (np.array),
            reward (float),
            possible actions in the next turn (np.array),
            game finished flag (0 or 1),
            action label (string)
        """

		reward, game_finished,clabel = self.take_action(action,self.current_player)

		return self.get_state_space(),reward,self.get_possible_actions(self.current_player),game_finished,clabel


	def get_possible_actions(self,player_num):
		"""
        Concatenates and return all possible actions of the given player.

        Herefore the function references given in self.action_array_names_dic are used. So it is a
        bit generic method.

        :param player_num:
        	Number of player
        :returns:
        	numpy array with possible actions (1's and 0's).
        """
		counter = 0
		possible_actions = []
		for action_set,function_ref in self.action_array_names_dic.items():
			if counter == 0:
				possible_actions = function_ref[0](player_num)*1
				counter+=1
			else:
				possible_actions = np.concatenate((possible_actions,function_ref[0](player_num)*1))
		if self.reward == 'cards':
			if self.action_counter >= 16:
				possible_actions = np.zeros(len(possible_actions))
				possible_actions[0] = 1
		return possible_actions

	def take_action(self,chosen_action_ind,player_num):
		"""
        Executes the action chosen by the RL algorithm

        Herefore the function references given in self.action_array_names_dic are used. So it is a
        bit generic method.
        :param action_index:
            Number of action.
        :param player_num:
        	Number of player
        """

		last_val = 0
		for key,val in self.act_dic.items():
			if chosen_action_ind < val:
				chosen_action_array_label = key # eg 'build_city'
				action_ind = chosen_action_ind-last_val
				self.action_array_names_dic[chosen_action_array_label][1](action_ind,player_num)
				break
			else:
				last_val = val
		self.count_up_action_counter()
		game_finished = 0

		reward = 0
		if self.reward == 'cards':

			finished = 0
			for i in range(5):
				if self.check_resource_available_on_pile(i) == False:
					finished = 1

			if finished:
				max_cards = np.max(np.sum(self.cards,axis=1))
				if max_cards == np.sum(self.cards,axis=1)[player_num-1]:
					return 1,1,' '
				else:
					return 0,1,' '
			else:
				return 0,0,' '

			#return np.sum(self.cards,axis=1)[player_num-1]/(0.001+np.sum(self.cards)),finished,' '

		if np.any(self.get_victory_points() >= self.needed_victory_points):
			game_finished = 1
			if np.argmax(self.get_victory_points())==player_num-1: #Player 1
				reward += 1
				
		if self.reward == 'victory_only':
			pass
		elif self.reward == 'building':
			if chosen_action_array_label in ['build_road']:
				reward += 0.3
			elif chosen_action_array_label in ['build_settlement','build_city']:
				reward += 0.6
			elif chosen_action_array_label in ['trade_4vs1']:
				reward -= 0.6
			else:
				reward -= 0.3
		return reward,game_finished,chosen_action_array_label




	def count_up_action_counter(self):
		"""
        Needed for the initial state when placing free settlements and roads.

        Maintains the correct players order and counts up the action_counter so
        that the game can continue as a normal game after initialization.
        """
		if self.action_counter%2==1 and self.action_counter <16:
			if self.action_counter < 7:
				self.current_player += 1
			elif self.action_counter == 7:
				pass
			elif self.action_counter < 15:
				self.current_player -= 1
		self.action_counter += 1

	def get_state_space(self):
		"""
        Returns the current state space

        Iterates through all subsets of the state space and concatenates their 
        specific state space representation
        
        :returns:
        	numpy array representing the current game state (1's and 0's).
        """
		counter = 0
		state_array = []
		for state_name,function_ref in self.state_array_dic.items():

			if counter == 0:
				state_array = function_ref()
				counter+=1
			else:
				state_array = np.concatenate((state_array,function_ref()))
		return state_array


	def roll_dice(self):
		"""
		Rolls the dice.

		If it is a 7, the robber shall be moved. Otherwise distribute the corresponding resources.
		"""
		number = random.choice(self.dices_results)
		if number == 7:
			# This could later be replaced by get_possible_actions_discard_resources, offering some discarding heuristics
			if self.reward == 'cards':
				return
			self.discard_resources()

			# Actually here we have to make sure that the next action taken by the player will be move_robber()
			if 'move_robber' in self.action_array_names_dic.keys():
				self.seven_rolled = 1
		else:
			self.distribute_resources(number)

	def move_robber(self,action_ind,player_num):
		"""
        Moves the robber to the tile with number action_ind

        :param player_num:
        	Number of player
        :param action_ind:
        	Number of player
        """
		self.seven_rolled = 0
		self.set_robber_position(action_ind)
		self.rob_player_state = 1
		if sum(self.get_possible_actions_rob_player(player_num))==0:
			self.rob_player_state = 0



	def distribute_resources(self,number):
		"""
		Distribute the resources according to the rules.

		:param number:
			Number rolled by the dices.
		"""
		crossings = self.crossings.get_crossings()
		buildings = self.crossings.get_building_state()

		# Go through all crossings
		for i in range(len(crossings)):
			# Checks if there is no building on the crossing
			if buildings[i] == 0 or buildings[i] == 9:
				continue
			# Iterate through all neighbouring tiles of the crossing
			for tile in crossings[i][1]:
				#Check if robber is on this tile
				if tile[2] == self.robber:
					continue
				# If the rolled number is a number chip on one of those crossings
				if tile[1]==number:

					self.add_resource(tile[0],(buildings[i]%4)-1)
					# If it is a town, add one more resource
					if buildings[i] > 4:
						self.add_resource(tile[0],(buildings[i]%4)-1)


	def add_resource(self,resource_type,player_num):
		"""
		Adds a resource of resource_type to the stack of cards of player player_num.

		Checks if resource is available at all.
		:param
			resource_type:
				Resource_type to be added.
			player_num:
				Player the resource is added to.
		"""
		if not self.check_resource_available_on_pile(resource_type):
			return
		self.cards[player_num,resource_type] += 1

	def check_resource_available_on_pile(self,resource_type):
		"""
		Checks if a certain resource type is still available to be drawn ba a player.

		:param resource_type:
			Resource_type to be checked.
		:return:
			bool
				True if the resource is available, False otherwise.
		"""
		if np.sum(self.cards[:,resource_type]) >= 19:
			return False
		return True

	def place_settlement(self,crossing_index,player_num):
		"""
		Places a settlement at the crossing index position.
        
        In the initial phase, placing the settlement is for free.
        The resources of the tile close to the second settlement are distributed
        according to the rules.

		:param crossing_index:
			Resource_type to be checked.
        :param player_num:
            Player number
		"""
		if self.action_counter >= 16:
			self.pay(player_num,buying_good='Settlement')
		elif self.action_counter >= 7: # second settlement placed
			self.distribute_resources_second_settlement(crossing_index,player_num)
		self.crossings.place_settlement(crossing_index,player_num)

		pass

	def distribute_resources_second_settlement(self,crossing_index,player_num):
		"""
		Distributes resources of second settlement according to the rules.

		:param crossing_index:
			Resource_type to be checked.
        :param player_num:
            Player number
		"""
		crossings = self.crossings.get_crossings()
		for tile in crossings[crossing_index][1]:
			if tile[1]!=7:
				self.add_resource(tile[0],player_num-1)

	def place_road(self,road_index,player_num):
		"""
		Places a road on the given road index.

		:param road_index:
			Index of the road.
        :param player_num:
            Player number
		"""
		if self.action_counter >= 16:
			self.pay(player_num,buying_good='Road')
		self.roads.place_road(road_index,player_num)
		pass

	def place_city(self,crossing_index,player_num):
		"""
		Places a city on the given crossing index.

		:param road_index:
			Index of the crossing.
        :param player_num:
            Player number
		"""
		self.pay(player_num,buying_good='City')
		self.crossings.place_city(crossing_index,player_num)
		pass

	def rob_player(self,robbed_player_index,player_num):
		"""
        Takes a random card from player robbed_player_index and passes it to player player_num.

		:param robbed_player_index:
            Player to be robbed.
        :param player_num:
            Number of the player.
        """
		# As get_action_rob_player swaps current players position with the first player,
		# it is reversed here
		if robbed_player_index == player_num-1:
			player_num = robbed_player_index + 1
			robbed_player_index = 0

		#print("Robbed player: ", robbed_player_index)
		rob_resource_index=np.random.choice(np.arange(5),1,p=self.cards[robbed_player_index,:]/sum(self.cards[robbed_player_index,:]))[0]
		self.cards[robbed_player_index,rob_resource_index]-=1
		self.cards[player_num-1,rob_resource_index]+=1
		self.rob_player_state = 0

		pass

	def get_possible_actions_build_settlement(self,player_num):
		"""
        Returns all locations where settlement can be placed by the given player.

        :param player_num:
            Number of the player.

        :return valid_crossings:
            list(int) list of all crossing indexes where a settlement is allowed to be placed
                by this player
        """

		# On Initialization
		if self.action_counter < 16 and self.action_counter%2==0:

			return self.crossings.get_building_state()==0


		valid_crossings = np.zeros(defines.NUM_CROSSINGS)
		if self.check_resources_available(player_num,'Settlement') == False:
			return valid_crossings
		if self.seven_rolled or self.rob_player_state:
			return valid_crossings

		# Find all crossings a road of this player is connected to
		crossings_connected_to_roads = np.unique(self.roads.get_roads()[self.roads.get_state()==player_num])

		# Iterate through all crossings a road of this player is connected to
		for crossing in crossings_connected_to_roads:
			valid_crossings[crossing] = 1 # assuming it is valid

			# Checks if there is a building on this crossing
			if self.crossings.building_state[crossing]>0:
				# If there is, the crossing is not valid for placing a settlement
				valid_crossings[crossing] = 0
				continue

			# Iterate through all crossings connected to this crossing
			for first_crossing in self.crossings.get_neighbouring_crossings()[crossing]:

				# Checks if there is a building on this crossing
				if self.crossings.building_state[first_crossing]>0 and self.crossings.building_state[first_crossing]<9:

					# If there is, the crossing is not valid for placing a settlement
					valid_crossings[crossing] = 0
					break

		# Returns the remaining valid crossings
		return valid_crossings

	def get_possible_actions_build_road(self,player_num):
		"""
		Returns a vector of zeros of length of amount of roads.
		with ones on indices where roads can be placed by the given player.

		:param player_num:
			Number of the player.
		"""
		# On initialization
		if self.action_counter < 16 and self.action_counter%2==1:

			# Find the settlement without a road close to it
			# Find indices of crossings with buildings
			ind_with_buildings = np.ravel(np.argwhere(self.crossings.get_building_state()==player_num))

			# Loop through all settlement of this player
			for crossing_index in ind_with_buildings:
				connected_roads = self.roads.get_state()
				# If no road connected to the settlement yet...
				if np.sum(connected_roads[self.crossings.connected_roads[crossing_index]])==0:
					final_arr = np.zeros(defines.NUM_EDGES)
					final_arr[self.crossings.connected_roads[crossing_index]]=1

					return final_arr

			return


		if self.check_resources_available(player_num, 'Road') == False:
			return np.zeros(defines.NUM_EDGES)
		if self.seven_rolled or self.rob_player_state:
			return np.zeros(defines.NUM_EDGES)


		# During normal game play...

		# Find all roads of player player_num and look for edges they are connected to
		conn_roads = np.array(self.roads.connected_roads)
		road_state = np.array(self.roads.road_state)
		list_conn = conn_roads[np.where(road_state==player_num)]

		# Convert those connected edges to an array of zeros with a one where a road can be placed
		final_list = []
		for lst in list_conn:
			final_list.extend(lst)
		final_arr = np.zeros(defines.NUM_EDGES)
		final_arr[final_list]=1

		# Exclude all occupied edges from the list of connected edges and return result
		return np.logical_and(np.logical_not(road_state),final_arr)

	def check_resources_available(self,player_num,buying_good):
		"""
        Check availability of resources for the specific buying good.

        :param player_num:
            Number of the player.

		:param	buying_good:
			One of the following: 'Road', 'Settlement', 'City', 'Development Card'
        """
		if buying_good=='Road':
			return (self.cards[player_num-1,:]>=np.array([0,0,0,1,1])).all()
		if buying_good=='Settlement':
			return (self.cards[player_num-1,:]>=np.array([1,1,0,1,1])).all()
		if buying_good=='City':
			return (self.cards[player_num-1,:]>=np.array([2,0,3,0,0])).all()
		if buying_good=='Development Card':
			return (self.cards[player_num-1,:]>=np.array([1,1,1,0,0])).all()

	def pay(self,player_num,buying_good):
		"""
        Reduces the cards of others player by the amount needed for the resources.

        :param player_num:
            Number of the player.

		:param buying_good:
			One of the following: 'Road', 'Settlement', 'City', 'Development Card'
        """
		if buying_good=='Road':
			self.cards[player_num-1,:]-=np.array([0,0,0,1,1])
		if buying_good=='Settlement':
			self.cards[player_num-1,:]-=np.array([1,1,0,1,1])
		if buying_good=='City':
			self.cards[player_num-1,:]-=np.array([2,0,3,0,0])
		if buying_good=='Development Card':
			self.cards[player_num-1,:]-=np.array([1,1,1,0,0])

	def get_possible_actions_build_city(self,player_num):
		"""
        Returns all locations where cities can be placed by the given player.

        :param player_num:
            Number of the player.

        :return valid_crossings:
            list(int) list of all crossing indexes where a city is allowed to be placed
                by this player
        """
		if self.seven_rolled or self.rob_player_state or self.action_counter < 16:
			return np.zeros(defines.NUM_CROSSINGS)
		if self.check_resources_available(player_num,'City') == False:
			return np.zeros(defines.NUM_CROSSINGS)
		valid_crossings = (self.crossings.get_building_state()==player_num)


		# Returns the remaining valid crossings
		return valid_crossings

	def set_robber_position(self,tile_number):
		"""
        Puts the robber on the specified tile.

        :param tile_number:
            Number of the tile between 0 and 18.
        """
		self.robber = tile_number

	def get_possible_action_move_robber(self,player_num):
		"""
        Returns all locations where the robber can be placed.

        :return robber_actions:
            np.array(binary), length 19, with a 1 representing a tile, the robber can be moved to and
            	a 0 where it can't.
        """
		if self.seven_rolled == 0 or self.action_counter < 16:
			return np.zeros(defines.NUM_TILES)

		robber_actions = np.ones(defines.NUM_TILES)
		# Robber has to move away from current position
		robber_actions[self.robber] = 0

		return robber_actions

	def get_robber_state(self):
		"""
        Returns the location of the robber in state representation.

        :return robber_state:
            np.array(binary), length 19, with a 1 representing the tile, where the robber is currently placed
            	and a 0 where it isn't.
        """
		robber_state = np.zeros(defines.NUM_TILES)
		robber_state[self.robber] = 1
		return robber_state


	def rob_person(self, player_num):
		"""
		Return list of players possible to rob resource from
		
		:return rob_players:
			np.array(binary), length 4, with 1 representing person which is robable, 0 not rob
		"""
		# Looking at tile number - surrounding crossings + if settlement or city is built on it
		building_state = self.crossings.get_building_state()
		crossing_index = []
		
		for n_tiles in self.crossings.neighbouring_tiles:
			for tile in n_tiles:
				if tile == self.robber:
					crossing_index.append(self.crossings.neighbouring_tiles.index(n_tiles))
		
		print("Crossing index ", crossing_index)
		possible_players = []
		for crossing in crossing_index:
				if building_state[crossing] != 0 and building_state[crossing] != 9:
					print("Crossing type ", building_state[crossing])
					possible_players.append(building_state[crossing])
		print("Possible players ", possible_players)			
		rob_players = [0,0,0,0]
		for player in possible_players:
				if player == 1 or player == 5:
					rob_players[0] = 1
				elif player == 2 or player == 6:
					rob_players[1] = 1
				elif player == 3 or player == 7:
					rob_players[2] = 1
				else:
					rob_players[3] = 1
		
		# self robbing not allowed
		rob_players[player_num-1] = 0
		return rob_players

	def get_possible_actions_rob_player(self,player_num):
		"""
		Return list of players possible to rob resource from. Checking if player even has resources is performed
		also now.

		:return rob_players:
			np.array(binary), length 4, with 1 representing person which is robable, 0 not rob
		"""
		if self.rob_player_state == 0 or self.action_counter < 16:
			return np.zeros(4)

		robber_crossings = self.crossings.get_crossings_per_tile()[self.robber,:]
		cross_state = self.crossings.building_state[robber_crossings]
		possible_players = []
		for crossing in cross_state:
			if crossing != 0 and crossing != 9:
				possible_players.append(crossing)
		rob_players = [0,0,0,0]
		for player in possible_players:
			if player == 1 or player == 5:
				if sum(self.cards[0,:])>0:
					rob_players[0] = 1
			elif player == 2 or player == 6:
				if sum(self.cards[1,:])>0:
					rob_players[1] = 1
			elif player == 3 or player == 7:
				if sum(self.cards[2,:])>0:
					rob_players[2] = 1
			elif player == 4 or player == 8:
				if sum(self.cards[3,:])>0:
					rob_players[3] = 1

		# self robbing not allowed
		# Putting the player who's turn is in the first position of the possible players to be robbed
		rob_players[player_num-1] = rob_players[0]
		rob_players[0] = 0

		return np.array(rob_players)

	def get_possible_actions_trade_bank(self,player_num):
		"""
		Return list of possible trades to be done with the bank (4 of the same resource vs 1 of free choice).

		:return trade_bank_arr:
			np.array(binary), length 20, with 1 representing a trade is possible.
			Representation:

			Grain against			Wool NOT against		Ore against
			Wool Ore Brick Wood		Grain Ore Brick Wood	Grain Wool Brick Wood
			[1 1 1 1                0 0 0 0 				1 1 1 1 ...]
		"""

		trade_bank_array = np.repeat(self.cards[player_num-1,:] >= 4,4)

		if self.seven_rolled or self.rob_player_state or self.action_counter < 16:
			return np.zeros(len(trade_bank_array))

		# vector with resource types received when trading according to trade_bank_array
		self.traded_resources = np.array([1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3])
		traded_resources_available = np.zeros(len(self.traded_resources))
		for i in range(5):
			traded_resources_available = np.logical_or(self.check_resource_available_on_pile(i)*(self.traded_resources==i),traded_resources_available)
		return np.logical_and(trade_bank_array,traded_resources_available)

	def get_possible_actions_trade_3vs1(self,player_num):
		"""
		Return list of possible trades to be done via 3vs1 trade. It is checked whether the player
		has a settlement on a 3vs1 port

		:return trade_3vs1_arr:
			vector of length 100 (5 Resource type x 20 combinations of remaining resources each)
			with a 1 where the set of three cards is on the players hand.
		"""
		possible_actions = [0]*100

		if self.seven_rolled or self.rob_player_state or self.action_counter < 16:
			return np.array(possible_actions)

		if not self.has_3vs1_port(player_num):
			return np.array(possible_actions)

		if sum(self.cards[player_num-1,:])<3:
			return np.array(possible_actions)

		# Contains resource indices of all resources the player has at least once
		non_zero_card_indices = np.nonzero(self.cards[player_num-1,:])[0]

		iter_cards = np.array([])
		# Builds all possible sets of three cards given the cards the player has
		#print(self.cards)
		for j in non_zero_card_indices:
			iter_cards =np.concatenate((iter_cards,np.repeat(j,self.cards[player_num-1,j])))

		card_sets = set(itertools.combinations(tuple(iter_cards),3))
		for j in range(len(self.trade_3vs1_list)):
			# Check if wanted resource is available on the pile at all..
			if not self.check_resource_available_on_pile(j):
				continue

			temp_list = sorted(self.trade_3vs1_list[j])
			for i in range(len(temp_list)):
				if temp_list[i] in card_sets:
					possible_actions[i+20*j] = 1
		return np.array(possible_actions)

	def create_possible_trade_sets_3vs1(self):
		"""
		Creates a list of length 5 (Number of resources) each containing a sorted set of sets.
		Each sorted set of sets index corresponds to the resource you want to obtain by trading.
		The inner sets contain 3 numbers corresponding to the resources you want to trade.

		Example:
			self.trade_3vs1_list = [{(1,1,1), (1,1,2), (1,1,3) ... (3,4,4), (4,4,4)},   <-- Set 0 corresponds to grain. Its inner sets therefore do not contain zeros
									{(0,0,0), (0,0,2), (0,0,3) ... (3,4,4), (4,4,4)},   <-- Set 1 corresponds to wool. Its inner sets therefore do not contain ones
		"""
		grain_3vs1 = set(sorted(set(itertools.combinations((1,1,1, 2,2,2, 3,3,3, 4,4,4), 3))))
		wool_3vs1 = set(sorted(set(itertools.combinations((0,0,0, 2,2,2, 3,3,3, 4,4,4), 3))))
		ore_3vs1= set(sorted(set(itertools.combinations(( 0,0,0,1,1,1, 3,3,3, 4,4,4), 3))))
		brick_3vs1 = set(sorted(set(itertools.combinations((0,0,0,1,1,1, 2,2,2,  4,4,4), 3))))
		wood_3vs1 = set(sorted(set(itertools.combinations((0,0,0,1,1,1, 2,2,2, 3,3,3 ), 3))))
		self.trade_3vs1_list = [grain_3vs1,wool_3vs1,ore_3vs1,brick_3vs1,wood_3vs1]

	def has_3vs1_port(self,player_num):
		"""
        Get information on whether the player has a settlement/city adjacent to a 3vs1 port.

        :param player_num:
            Number of the player.
        :return arr :
        	Array of length 5 (resource types) with a 1 indicating that the player has a 3vs1 port.
        """
		building_state =self.crossings.get_building_state()*(self.crossings.get_building_state()<9)
		settlement_state = (building_state==(player_num))+(building_state==(player_num+4))

		harbour_state = list(zip(*self.crossings.get_crossings()))[2]
		return np.any(((harbour_state*settlement_state)==defines.PORT_ANY))

	def has_2vs1_port(self,player_num):
		"""
        Get information on whether the player has a settlement/city adjacent to a 2vs1 port.

        :param player_num:
            Number of the player.
        :return arr :
        	Array of length 5 (resource types) with a 1 indicating that the player has such a 2vs1 port.
        """
		building_state =self.crossings.get_building_state()*(self.crossings.get_building_state()<9)
		settlement_state = (building_state==(player_num))+(building_state==(player_num+4))
		harbour_state = list(zip(*self.crossings.get_crossings()))[2]
		has_2vs1_port = []
		has_2vs1_port.append(np.any(((harbour_state*settlement_state)==defines.PORT_FIELDS)))
		has_2vs1_port.append(np.any(((harbour_state*settlement_state)==defines.PORT_PASTURE)))
		has_2vs1_port.append(np.any(((harbour_state*settlement_state)==defines.PORT_MOUNTAINS)))
		has_2vs1_port.append(np.any(((harbour_state*settlement_state)==defines.PORT_HILLS)))
		has_2vs1_port.append(np.any(((harbour_state*settlement_state)==defines.PORT_FOREST)))
		return np.array(has_2vs1_port)

	def get_possible_actions_trade_2vs1(self,player_num):
		"""
		Return list of possible trades to be done with a port, if the (4 of the same resource vs 1 of free choice).

		:return array:
			np.array(binary), length 20, with 1 representing a trade is possible.
			Representation:

			Grain against			Wool NOT against		Ore against
			Wool Ore Brick Wood		Grain Ore Brick Wood	Grain Wool Brick Wood
			[1 1 1 1                0 0 0 0 				1 1 1 1 ...]
		"""

		if self.seven_rolled or self.rob_player_state or self.action_counter < 16:
			return np.zeros(20)

		has_2vs1_port = self.has_2vs1_port(player_num)
		port_and_cards_available = has_2vs1_port * (self.cards[player_num-1,:]>=2)

		# Check if resources are available on the pile
		self.traded_resources = np.array([1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3])
		traded_resources_available = np.zeros(len(self.traded_resources))
		for i in range(5):
			traded_resources_available = np.logical_or(self.check_resource_available_on_pile(i)*(self.traded_resources==i),traded_resources_available)

		return np.logical_and(np.repeat(port_and_cards_available,4),traded_resources_available)

	def discard_resources(self):
		"""
		Simple heuristic for discarding cards.

		If the player has more than 7 cards he discards half of them always discarding the resource type
		he has most of until he reaches half of his cards (+1 if uneven)
		"""
		for i in range(4):
			player_cards = self.cards[i,:]
			if sum(player_cards) > 7:
				num_discarded_cards = int(sum(player_cards)/2)
				for j in range(num_discarded_cards):
					player_cards[np.argmax(player_cards)]-=1
			self.cards[i,:] = player_cards

	def create_possible_actions_dictionary(self):
		"""
		According to initialization of the game, only certain action spaces will be added to this dictionary.
		They will be added in the order of appearance in  self.action_array_names_dic.
		Here the first index of the next action space is stored with each named action space.

		Example:
			{'build_road':72,
			 'build_settlement':126
			 ...}
		"""
		act_dic = {}
		last_length = 0
		for action_set,function_ref in self.action_array_names_dic.items():
			act_dic[action_set] = len(function_ref[0](1)) + last_length
			last_length = act_dic[action_set]

		self.act_dic = act_dic

	def trade_bank(self,action_index,player_num):
		"""
        Executes the action trade with bank for a certain player and a certain action index (corresponding to
        the get_possible_actions_trade_bank function)

        :param action_index:
            Number of action.
        :param player_num:
        	Player Number.
        """
		player_num-=1
		traded_resource = int(action_index/4)
		self.cards[player_num,traded_resource]-=4
		self.cards[player_num,self.traded_resources[action_index]]+=+1

	def trade_3vs1(self,action_index,player_num):
		"""
        Executes the action trade 3vs1 for a certain player and a certain action index (corresponding to
        the get_possible_actions_trade_3vs1 function)

        :param action_index:
            Number of action.
        :param player_num:
        	Player Number.
        """
		player_num-=1
		traded_resource = int(action_index/20) # the resource you the player will obtain
		self.cards[player_num,traded_resource]+=1
		given_tuple = sorted(self.trade_3vs1_list[traded_resource])[action_index%20]
		for resource in given_tuple:
			self.cards[player_num,resource]-=1


	def trade_2vs1(self,action_index,player_num):
		"""
        Executes the action trade 2vs1 for a certain player and a certain action index (corresponding to
        the get_possible_actions_trade_2vs1 function)

        :param action_index:
            Number of action.
        :param player_num:
        	Player Number.
        """
		player_num-=1
		traded_resource = int(action_index/4)
		self.cards[player_num,traded_resource]-=2
		self.cards[player_num,self.traded_resources[action_index]]+=1

	def get_victory_points(self):
		"""
        Calculates the current victory points for each player

        :return numpy array:
        	Contains the current victory points of each player
        """
		vp = []
		buildingstate = self.crossings.get_building_state()
		#devcardstate = self.get_dev_card_state()
		for i in range(4):
			player_num = i+1
			vp.append(sum(buildingstate == player_num)
					+ 2*sum(buildingstate == (player_num+4))
					+ self.dev_cards_discovered[player_num-1][defines.DEV_VICTORYPOINT])

		return np.array(vp)

	def get_dev_card(self, player_num):

		#check if stack is empty

		if self.dev_card_stack:
			#TBD resources need to be checked if available
			self.pay(player_num, buying_good='Development Card')

			drawn_card = self.dev_card_stack[0]
			self.dev_card_stack.remove(drawn_card)
			self.dev_cards[player_num][drawn_card] += 1

	def dev_playable(self):
		#makes cards drawn last round playable
		#difference from round before and now -> playable cards

		pass

	# vllt alles eher in eine fkt hauen

	def dev_knight(self, robbed_player_index, action_ind, player_num):

		if self.dev_cards_playable[player_num][defines.DEV_KNIGHT]:

			self.dev_cards[player_num][defines.DEV_KNIGHT] -= 1
			self.dev_cards_playable[player_num][defines.DEV_KNIGHT] -= 1
			self.dev_cards_discovered[player_num][defines.DEV_KNIGHT] += 1

			self.move_robber(action_ind, player_num)
			self.rob_player(robbed_player_index, player_num+1)
		pass

	def dev_vic(self, player_num):
		if self.dev_cards_playable[player_num][defines.DEV_VICTORYPOINT]:

			self.dev_cards[player_num][defines.DEV_VICTORYPOINT] -= 1
			self.dev_cards_playable[player_num][defines.DEV_VICTORYPOINT] -= 1
			self.dev_cards_discovered[player_num][defines.DEV_VICTORYPOINT] += 1

		pass

	def dev_yearofplenty(self, player_num):
		if self.dev_cards_playable[player_num][defines.DEV_YEAROFPLENTY]:

			self.dev_cards[player_num][defines.DEV_YEAROFPLENTY] -= 1
			self.dev_cards_playable[player_num][defines.DEV_YEAROFPLENTY] -= 1
			self.dev_cards_discovered[player_num][defines.DEV_YEAROFPLENTY] += 1

		pass

	def dev_monopoly(self, player_num):
		if self.dev_cards_playable[player_num][defines.DEV_MONOPOLY]:

			self.dev_cards[player_num][defines.DEV_MONOPOLY] -= 1
			self.dev_cards_playable[player_num][defines.DEV_MONOPOLY] -= 1
			self.dev_cards_discovered[player_num][defines.DEV_MONOPOLY] += 1
		
		pass

	def dev_road(self, road_index, player_num):
		if self.dev_cards_playable[player_num][defines.DEV_ROADBUILDING]:

			self.dev_cards[player_num][defines.DEV_ROADBUILDING] -= 1
			self.dev_cards_playable[player_num][defines.DEV_ROADBUILDING] -= 1
			self.dev_cards_discovered[player_num][defines.DEV_ROADBUILDING]+= 1

		for _ in range(2):
			self.roads.place_road(road_index, player_num)
		pass

### State Space Getters
	def get_state_space_tiles(self):
		# Returns a flattened One-hot representation of the resources for each tile
		return self.tile_space

	def get_state_space_numbers(self):
		return self.number_space

	def create_number_space(self):
		num_prob_dist = np.array([1,2,3,4,5,6,5,4,3,2,1])/4 # 4 instead of 36 in order to have inputs in similar numeric ranges
		resource,number=zip(*self.tiles.get_tiles())

		self.number_space = num_prob_dist[np.array(number)-2]
		self.number_space[np.argmax(self.number_space)]=0

	def create_tile_space(self):
		resources,number=zip(*self.tiles.get_tiles())
		n_highest_value = np.max(resources) + 1
		self.tile_space = np.ravel(np.eye(n_highest_value)[np.array(resources)].T*self.number_space)

	def get_state_space_ports(self):
		return self.harbour_space

	def create_port_space(self):
		harbour_state = self.tiles.get_harbour_state()
		n_highest_value = np.max(harbour_state) # as they are numbered from 1-6
		self.harbour_space = np.ravel(np.eye(n_highest_value)[np.array(harbour_state)-1])

	def get_state_space_buildings(self):
		building_state = self.crossings.get_building_state().copy()
		building_state[np.where(building_state==9)]=0
		if not self.current_player == 1:
			swap_1 = ((building_state==1)+(building_state==5))*(self.current_player-1)
			swap_player = ((building_state==self.current_player)+(building_state==(self.current_player+4)))*(1-self.current_player)
			building_state = building_state + swap_1 + swap_player
		n_highest_value = 9

		return np.ravel(np.eye(n_highest_value)[building_state])

	def get_state_space_roads(self):
		road_state = self.roads.get_state()
		if not self.current_player == 1:
			swap_1 = (road_state==1)*(self.current_player-1)
			swap_player = (road_state==self.current_player)*(1-self.current_player)
			road_state = road_state + swap_1 + swap_player
		n_highest_value = 4 # 1,2,3,4
		road_state-=1
		return np.ravel(np.eye(n_highest_value)[road_state])

	def get_state_space_robber(self):
		robber_state = np.zeros(defines.NUM_TILES)
		robber_state[self.robber]=1
		return robber_state

	def get_state_space_cards(self):
		card_state = self.cards.copy()
		if not self.current_player == 1:
			card_state[[0, self.current_player-1]] = card_state[[self.current_player-1, 0]]
		return np.ravel(card_state)/4 # divided by 4 in order to have inputs in similar ranges






