#!/usr/bin/env python
# coding: utf-8

from agent.distributedtraining import DistributedTraining
from agent.traincatan import TrainCatan

# -----TO DO -------- Loop over for 4 players

train = None
d = None
if __name__ == "__main__":

    d = DistributedTraining('cards',{'learning_rate':[0.1],'list_num_neurons':[(50,),(50,50),(50,50,50),(50,50,50,50)],'random_init':[True],
                                                 'random_shuffle_training_players_':[False,True],'needed_victory_points':[3],
                                             'replace_target_iter':[200],'verbose':[False],'memory_size':[100000],
                                                      'sigmoid_001_099_borders' : [(-1000,10000)],
                                             'batch_size':[1024],'learning_rate_start_decay':[5000],'num_games' : [20000,20000,20000,20000],
                                             'reward':['cards'],'learning_rate_decay_factor':[0.9998]
                                             })
    """
    train = TrainCatan(needed_victory_points=3, list_num_neurons=(50,), batch_size=1024, output_graph=False,show_cards_statistic=True,
                       learning_rate=0.1, random_init=True,random_shuffle_training_players=False,reward='cards',
                       memory_size=100000, sigmoid_001_099_borders=(-1000, 10000), replace_target_iter=200,reward_decay=1,num_games=100000,
                       learning_rate_decay_factor=0.9998,learning_rate_start_decay=10000)
    # train.RL.load_model('learningrate13.data-00000-of-00001')
    # train.RL.epsilon = 1
    train.start_training()
    """










