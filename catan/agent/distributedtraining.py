
import subprocess
import os
import sys
import time
import pickle
import datetime
from shutil import copyfile
from matplotlib import pyplot as plt

from agent.traincatan import TrainCatan
from gcloud.gcloudinstance import GcloudInstance
from itertools import product

INSTANCES_FOLDER = 'instance_parameters'
GCLOUDEXECUTABLE = '/home/angelo/Downloads/google-cloud-sdk/bin/gcloud'


ZONES = ('europe-west1-b',#'europe-west1-c','europe-west1-d',
         'europe-west2-a',#'europe-west2-b','europe-west2-c',
         'europe-west3-a',#'europe-west3-b','europe-west3-c',
         'europe-west4-a',#'europe-west4-b','europe-west4-c',
         'europe-west6-a'#'europe-west6-b','europe-west6-c')
         )
class DistributedTraining():
    """Class for distributing the training on several gcloud instances.

    Initializes and creates an amount of gcloud instances corresponding to the
    product of the length of the lists (values) of all keys in param_dic.
    E.g. param_dic = {'x':[1,2,3],'y':['a','b','c'],z:[4,5]} would result in
    3 x 3 x 2 = 36 instances.

    Attributes:
        instances_name_base (str): Name of the instances in the gcloud environment.
        param_dic (dict(str,str)): Contains arguments of the TrainCatan Class as key
                                   and a list of values this argument shall be set as value.

    """

    def __init__(self,instances_name_base,param_dic):

        self.g_cloud_instances = []

        assert type(param_dic) == dict
        self.t = TrainCatan()
        for param in param_dic:
            if not hasattr(self.t, param):
                print(''.join(['TrainCatan has no attribute named ',param]))
                sys.exit()


        list_combinations = [dict(zip(param_dic, v)) for v in product(*param_dic.values())]
        counter = 0
        zone_counter = 0
        for param_combination in list_combinations:

            instance_name = instances_name_base+str(counter)
            self.create_startup_script(instance_name,param_combination)
            self.g_cloud_instances.append(GcloudInstance(instance_name,param_combination,ZONES[zone_counter]))
            self.g_cloud_instances[-1].start_instance()
            counter+=1
            if counter%8 == 0:
                zone_counter += 1

        self.outstanding_instance_files = [instance.instance_name for instance in self.g_cloud_instances]

    def delete_instances(self):
        """Deletes all gcloud instances created on initialization of a DistributedTraining instance.
        """
        for instance in self.g_cloud_instances:
            instance.remove_instance()

    def create_startup_script(self,instance_name,param_combination):
        """Appends command line arguments to the startup script.

        The following command line arguments will be parsed tot he startup-script-template.

        instance_name param_name1 param_value1 param_name2 param_value2 ...

        Args:
            instance_name: The complete name given to the instance
            param_combination: The combination of parameters passed to this specific instance
        """
        param_value_string = ''.join([' ',instance_name])
        for param in param_combination:
            param_value_string = ''.join([param_value_string,' ',param,' '])

            if isinstance(param_combination[param],tuple):
                param_value_string = ''.join([param_value_string,str('\''),str(param_combination[param]),str('\'')])
            else:
                param_value_string = ''.join([param_value_string,str(param_combination[param])])

        copyfile('../startup_script_template.sh', '../startup_script.sh')
        with open('../startup_script.sh','rb+') as f: #open in binary mode
            f.seek(-1,2) #move to last char of file
            f.write(param_value_string.encode())
            f.close()
        print(param_value_string)

    def request_hyperparameter_files_from_instances(self):
        """
        Request all instance information including victory curves from the instance.

        If available, the files are sent to the host machine in the folder specified
        by the variable INSTANCES_FOLDER.
        The syntax for the gcloud command which is executed can be found under
        https://cloud.google.com/sdk/gcloud/reference/compute/scp.
        """

        if not os.path.exists(INSTANCES_FOLDER):
            os.mkdir(INSTANCES_FOLDER)

        while(len(self.outstanding_instance_files) > 0):
            time.sleep(30)
            print(''.join([str(self.outstanding_instance_files),' still not obtained.']))
            for instance in self.g_cloud_instances:

                if instance.instance_name in self.outstanding_instance_files:
                    dst_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),''.join([INSTANCES_FOLDER,'/',instance.instance_name]))
                    src_file = ''.join(['hyperparameters/',instance.instance_name,'/',instance.instance_name])
                    if self.scp_request(instance,src_file,dst_file) == 0:
                        print(''.join([instance.instance_name,' finished.']))
                        self.outstanding_instance_files.remove(instance.instance_name)
                        self.get_model(instance.instance_name)
                        instance.remove_instance()
                    else:
                        episode_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'episodes')
                        self.scp_request(instance,'episodes',episode_path)
                        if os.path.isfile(episode_path):
                            with open(episode_path,'r') as f:
                                print(''.join([str(instance.instance_name),' current episode: ',f.read()]))

                if len(self.outstanding_instance_files) == 0:
                    print('Obtained all hyperparameter files.')
                    break


    def scp_request(self,instance,src_filename,dst_filename):
        """
        Request file from instance. The relative path is the folder /catan/NI-Project---RL---Catan/Game_API/.

        :param instance: name of the instance
        :param filename: filename to be requested
        :return: error code, 0: scp transfer worked just fine
        """
        return_value = subprocess.call(["gcloud", "compute" ,"scp","--zone",instance.zone, ''.join([instance.instance_name,':/catan/NI-Project---RL---Catan/Game_API/',src_filename])
                                           ,dst_filename],
                                       executable=GCLOUDEXECUTABLE)
        return return_value

    def show_instances_graphs(self):
        """Displays figures of all trainings that happened on gcloud instances.

        """
        plot_counter = 0
        subplotcounter =0
        subplotnum = 321
        fig = plt.figure(plot_counter)
        for instance in self.g_cloud_instances:
            ax = fig.add_subplot(subplotnum)
            ax.title.set_text(instance.instance_name)
            t_i = self.load_hyperparameters(instance.instance_name)
            if t_i is None:
                continue
            t_i.autosave = False
            t_i.init_online_plot(make_new_figure = False)
            t_i.plot_statistics_online(t_i.victories,t_i.epsilons,t_i.cards,t_i.one_of_training_instances_wins,t_i.learning_rates,t_i.plot_interval)
            subplotnum+=1
            subplotcounter+=1
            if subplotcounter%6 ==0:
                plot_counter+=1
                fig = plt.figure(plot_counter)
                subplotnum = 321



    def load_hyperparameters(self,filename):
        """Loads an instantiation of a TrainCatan object saved in a pickle file.

        The file has to be located in the ./INSTANCES_FOLDER .

        Args:
            filename: Filename to be loaded.
        """
        if not os.path.isfile(''.join([INSTANCES_FOLDER,'/',filename])):
            return
        f = open(''.join([INSTANCES_FOLDER,'/',filename]), 'rb')
        return pickle.load(f)

    def get_model(self,instance_name):

        for instance in self.g_cloud_instances:
            dst_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),''.join(['models/',instance.instance_name]))
            if instance.instance_name == instance_name:
                self.scp_request(instance,''.join(['models/',str(datetime.date.today()),'.data-00000-of-00001']),
                                                  ''.join([dst_file,'.data-00000-of-00001']))
                self.scp_request(instance,''.join(['models/',str(datetime.date.today()),'.index']),
                                 ''.join([dst_file,'.index']))
                self.scp_request(instance,''.join(['models/',str(datetime.date.today()),'.meta']),
                                 ''.join([dst_file,'.meta']))
                self.scp_request(instance,''.join(['models/','checkpoint']),
                                 'checkpoint')


