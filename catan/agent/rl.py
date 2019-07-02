"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np

import tensorflow as tf
from scipy.special import softmax
import os

MODELFOLDER = '/models/'

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,  # Total actions
            n_features, # Total features/states
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,  # Num of iterations after which params of Q_Eval and Q_Target will be exchanged
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            softmax_choice=False,
            list_num_neurons= (50,50),
            activation_function = 'relu',
            loss_function = 'mse',
            optimizer_function = 'gradient',
            dropout_prob = 0.2
    ):
        # Initialize the params passed from run_this file
        self.summaries_dir = 'Summaries'
        self.summaries = []
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = tf.Variable(learning_rate, trainable=False,dtype=tf.float64)
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.softmax_choice = softmax_choice
        self.epsilon = 0 if e_greedy_increment is not None else e_greedy
        self.learn_step_counter = 0
        self.dropout_prob = dropout_prob
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.list_num_neurons = list_num_neurons
        # consist of [target_net, evaluate_net]
        self.sess = tf.InteractiveSession()

        self.get_NN_functions(loss_function,activation_function,optimizer_function)
        self._build_net(self.loss_function,self.activation_function,self.optimizer_function)
        # get_collection return the list of values associated with target_net_params & eval_net_params (refer to  _build_net)
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        tau = 0.0001

        self.replace_hard_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.replace_soft_target_op = [v_t.assign(v_t * (1. - tau) + v * tau) for v_t, v in zip(t_params, e_params)]

        self.output_graph = output_graph

        if self.output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        global_step_tensor = tf.Variable(0,trainable=False,name='global_step')
        self.sess.run(global_step_tensor.initializer)
        self.cost_his = []

    def get_NN_functions(self,loss_function,activation_function,optimizer_function):
        if activation_function == 'relu':
            self.activation_function = tf.nn.relu
        elif activation_function == 'tanh':
            self.activation_function = tf.nn.tanh
        else:
            self.activation_function = tf.nn.relu

        if loss_function == 'mse':
            self.loss_function = tf.losses.mean_squared_error
        elif loss_function == 'huber':
            self.loss_function = tf.losses.huber_loss
        else:
            self.loss_function = tf.losses.mean_squared_error

        if optimizer_function == 'gradient':
            self.optimizer_function = tf.train.GradientDescentOptimizer
        elif optimizer_function == 'RMS':
            self.optimizer_function = tf.train.RMSPropOptimizer
        elif optimizer_function == 'adam':
            self.optimizer_function = tf.train.AdamOptimizer
        else:
            self.optimizer_function = tf.train.GradientDescentOptimizer



    def _build_net(self,loss_function=tf.losses.mean_squared_error,activation_function=tf.nn.relu,
                   optimizer_function=tf.train.GradientDescentOptimizer,with_bias=True):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        amount_layers = len(self.list_num_neurons)
        self.list_num_neurons = list(self.list_num_neurons)
        self.list_num_neurons.append(self.n_actions)
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, list_num_neurons, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.list_num_neurons, \
                tf.contrib.layers.xavier_initializer(), tf.random_uniform_initializer(-0.1,0.1)  # config of layers


            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                self.w1 = tf.get_variable('w1', [self.n_features, list_num_neurons[0]], initializer=w_initializer, collections=c_names)
                if with_bias:
                    self.b1 = tf.get_variable('b1', [1, list_num_neurons[0]], initializer=b_initializer, collections=c_names)
                    self.l1 = activation_function(tf.matmul(self.s, self.w1) + self.b1)
                else:
                    self.l1 = activation_function(tf.matmul(self.s, self.w1))
                self.dropout1 = tf.nn.dropout(self.l1, rate = self.dropout_prob)
            for layer_num in range(len(list_num_neurons)-1):

                # hidden layers. collections is used later when assign to target net
                with tf.variable_scope(''.join(['l',str(layer_num+2)])):
                    setattr(self,''.join(['w',str(layer_num+2)]) , tf.get_variable(''.join(['w',str(layer_num+2)]), [list_num_neurons[layer_num], list_num_neurons[layer_num+1]],
                                                                              initializer=w_initializer, collections=c_names))
                    if with_bias:
                        setattr(self,''.join(['b',str(layer_num+2)]) , tf.get_variable(''.join(['b',str(layer_num+2)]), [1, list_num_neurons[layer_num+1]],
                                                                              initializer=w_initializer, collections=c_names))
                        setattr(self,''.join(['l',str(layer_num+2)]) , activation_function(tf.matmul(getattr(self, ''.join(['dropout',str(layer_num+1)])),
                                                                                           getattr(self,''.join(['w',str(layer_num+2)])))+ getattr(self,''.join(['b',str(layer_num+2)]))))
                    else:
                        setattr(self,''.join(['l',str(layer_num+2)]) , activation_function(tf.matmul(getattr(self, ''.join(['dropout',str(layer_num+1)])),
                                                                                                     getattr(self,''.join(['w',str(layer_num+2)])))))
                    setattr(self,''.join(['dropout',str(layer_num+2)]) , tf.nn.dropout(getattr(self, ''.join(['l',str(layer_num+2)])),rate = self.dropout_prob))

            self.q_eval = getattr(self,''.join(['dropout',str(amount_layers+1)]))


        with tf.variable_scope('loss'):
            self.loss = loss_function(self.q_target, self.q_eval)
            

        with tf.variable_scope('train') as self.train_var:
            optimizer =  optimizer_function(self.lr)
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            self._train_op = optimizer.minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('lt1'):
                self.wt1 = tf.get_variable('wt1', [self.n_features, list_num_neurons[0]], initializer=w_initializer, collections=c_names)
                if with_bias:
                    self.bt1 = tf.get_variable('bt1', [1, list_num_neurons[0]], initializer=b_initializer, collections=c_names)
                    self.lt1 = activation_function(tf.matmul(self.s_, self.wt1) + self.b1)
                else:
                    self.lt1 = activation_function(tf.matmul(self.s_, self.wt1))# + self.b1)
                self.dropoutt1 = tf.nn.dropout(self.lt1, rate = self.dropout_prob)

            for layer_num in range(len(list_num_neurons)-1):

                # hidden layers. collections is used later when assign to target net
                with tf.variable_scope(''.join(['lt',str(layer_num+2)])):
                    setattr(self,''.join(['wt',str(layer_num+2)]) , tf.get_variable(''.join(['wt',str(layer_num+2)]), [list_num_neurons[layer_num], list_num_neurons[layer_num+1]],
                                                                                   initializer=w_initializer, collections=c_names))
                    if with_bias:
                        setattr(self,''.join(['bt',str(layer_num+2)]) , tf.get_variable(''.join(['b',str(layer_num+2)]), [1, list_num_neurons[layer_num+1]],
                                                                                   initializer=w_initializer, collections=c_names))
                        setattr(self,''.join(['lt',str(layer_num+2)]) , activation_function(tf.matmul(getattr(self, ''.join(['dropoutt',str(layer_num+1)])),
                                                                                           getattr(self,''.join(['wt',str(layer_num+2)])))+ getattr(self,''.join(['b',str(layer_num+2)]))))
                    else:
                        setattr(self,''.join(['lt',str(layer_num+2)]) , activation_function(tf.matmul(getattr(self, ''.join(['dropoutt',str(layer_num+1)])),
                                                                                                     getattr(self,''.join(['wt',str(layer_num+2)])))))
                    setattr(self,''.join(['dropoutt',str(layer_num+2)]) , tf.nn.dropout(getattr(self, ''.join(['lt',str(layer_num+2)])),rate = self.dropout_prob))
                    
            self.q_next = getattr(self,''.join(['dropoutt',str(amount_layers+1)]))


        # Name scope allows you to group various summaries together
        # Summaries having the same name_scope will be displayed on the same row
        with tf.name_scope('performance'):
            # Summaries need to be displayed
            # Whenever you need to record the loss, feed the mean loss to this placeholder
            self.tf_loss_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary')
            # Create a scalar summary object for the loss so it can be displayed
            self.tf_loss_summary = tf.summary.scalar('loss', self.tf_loss_ph)

        for g,v in self.grads_and_vars:
            if ''.join(['w',str(amount_layers+1)]) in v.name :
                with tf.name_scope('gradients'):
                    self.tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
                    self.tf_gradnorm_summary = tf.summary.scalar('grad_norm', self.tf_last_grad_norm)
                    break

        # Summaries need to be displayed
        # Create a summary for each weight bias in each layer
        all_summaries = []

        with tf.variable_scope('eval_net',reuse=True):
            for lid in range(len(list_num_neurons)):
                layer_name = ''.join(['l',str(lid+1)])
                with tf.name_scope(layer_name+'_hist'):
                    with tf.variable_scope(layer_name,reuse=True):
                        w = tf.get_variable(''.join(['w',str(lid+1)]))#, tf.get_variable(''.join(['b',str(lid+1)]))
                        #w,b = tf.get_variable(''.join(['w',str(lid+1)],tf.get_variable(''.join(['b',str(lid+1)]))

                        # Create a scalar summary object for the loss so it can be displayed
                        tf_w_hist = tf.summary.histogram('weights_hist', tf.reshape(w,[-1]))
                        #tf_b_hist = tf.summary.histogram('bias_hist', b)
                        all_summaries.extend([tf_w_hist])#, tf_b_hist])

        with tf.variable_scope('target_net',reuse=True):
            for lid in range(len(list_num_neurons)):
                layer_name = ''.join(['lt',str(lid+1)])
                with tf.name_scope(layer_name+'_hist'):
                    with tf.variable_scope(layer_name,reuse=True):
                        w = tf.get_variable(''.join(['wt',str(lid+1)]))#, tf.get_variable(''.join(['b',str(lid+1)]))
                        #w,b = tf.get_variable(''.join(['w',str(lid+1)],tf.get_variable(''.join(['b',str(lid+1)]))

                        # Create a scalar summary object for the loss so it can be displayed
                        tf_w_hist = tf.summary.histogram('weights_hist', tf.reshape(w,[-1]))
                        #tf_b_hist = tf.summary.histogram('bias_hist', b)
                        all_summaries.extend([tf_w_hist])#, tf_b_hist])

        # Merge all parameter histogram summaries together
        self.tf_param_summaries = tf.summary.merge(all_summaries)


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        """
        if ((self.memory_counter*4) % self.memory_size == 0) and (self.memory_counter > 1000):
            for i in range(1000):

                self.learn()
                self.learn_step_counter -= 1
        """

        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation, possible_actions):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation}).flatten()

            possible_action_indices = np.where(possible_actions==1)[0]
            q_poss = actions_value[possible_action_indices]
            #if self.learn_step_counter % self.replace_target_iter == 0:
            # print('Possible action values: '+str(q_poss))
            if self.softmax_choice:

                q_softmax = softmax(q_poss)
                return possible_action_indices[np.random.choice(len(q_softmax),1,p=q_softmax)[0]]

            action = possible_action_indices[np.argmax(q_poss)]
        else:
            action = np.random.choice(np.where(possible_actions==1)[0])
        return action

    def learn(self):
        #
        # check to replace target parameters
        #if self.learn_step_counter % self.replace_target_iter == 0:
        self.sess.run(self.replace_soft_target_op)

            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]
#Train the eval net wrt to the batch memory
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)


        # train eval network
        summary,self.cost,gn_summ,wb_summ = self.sess.run([self._train_op, self.loss,self.tf_gradnorm_summary,self.tf_param_summaries],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target
                                                })
        if self.learn_step_counter % self.replace_target_iter == 0:
            summary = self.sess.run(self.tf_loss_summary,feed_dict={self.tf_loss_ph:self.cost})
            if self.output_graph:
                # Returning Errors
                self.writer.add_summary(summary, self.learn_step_counter)
                self.writer.add_summary(gn_summ,self.learn_step_counter)
                self.writer.add_summary(wb_summ,self.learn_step_counter)

        self.cost_his.append(self.cost)

        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure(0)
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def get_num_total_trainable_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value

            total_parameters += variable_parameters
        return total_parameters


    def variable_summaries(self,var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def save_current_model(self,model_name):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess,os.getcwd()+MODELFOLDER +model_name)

    def save_model_interval(self,model_name,global_step=1000):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess,os.getcwd()+MODELFOLDER +model_name,global_step=global_step)

    def load_model(self,model_name):
        self.saver = tf.train.import_meta_graph(os.getcwd()+MODELFOLDER+model_name)
        self.saver.restore(self.sess,tf.train.latest_checkpoint(os.getcwd()+MODELFOLDER+'./'))



