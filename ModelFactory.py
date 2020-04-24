from WeightAgnosticNetwork import CustomModel
from MLPModel import MLP
import tensorflow as tf
import gym
import numpy as np

from gym_cartpole_swingup.envs import CartPoleSwingUpEnv
from copy import deepcopy
from tensorboard import main as tb


BIAS_VALUE = 1
COMPLEXITY_PENALTY = 3
SHARED_WEIGHTS_ = {-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2}
SHARED_WEIGHTS = [-2, -1, -0.5, 0.5, 1, 2]

EVOLUTION_NAME_CHOICES = ['add', 'insert', 'change']

EVOLUTION_NAMES = {'add': MLP.addConnection, 'insert': MLP.insertNode, 'change': MLP.changeActivation}

# class for management of models, changing the modelplan and comparing with other models
class ModelWrapper:
    def __init__(self):
        self.tf_model = None
        self.model_plan = None
        self.max_reward = 0
        self.average_reward = 0
        self.best_weight = 1

    # update the tensorflow model with one based on the modelplan
    def updateTFmodel(self):
        self.tf_model = CustomModel(self.model_plan, -1)

    # make changes to modelplan based on topology search operators
    def evolveModel(self, method):
        EVOLUTION_NAMES[method](self.model_plan)

    @property
    def modelMetrik2(self):
        return (self.max_reward + self.average_reward) / 2

    @property
    def modelMetrik(self):
        return self.average_reward - (self.model_plan.complexity * COMPLEXITY_PENALTY)

    def __str__(self):
        print("")
        print("Scores for : {}".format(id(self.tf_model)))
        print("Maximum Reward : {}".format(self.max_reward))
        print("Average Reward : {}".format(self.average_reward))
        print("Average+Complexity Metrik : {}".format(self.modelMetrik))
        print("Average+Maximum Metrik : {}".format(self.modelMetrik2))
        return ""




# function to create the initial models with wrapper
def createModels(model_mask, n_inputs, n_outputs):
    models = []
    for model_cons in model_mask:
        wrapper = ModelWrapper()
        wrapper.model_plan = MLP(n_inputs, n_outputs, model_cons)
        wrapper.updateTFmodel()
        models.append(wrapper)
    return models


# run model in environment and calculate reward of given steps
def run_model(env, model, render=False, steps=1000):
    last_reward = 0
    observation = env.reset()
    for i in range(steps):
        new_observation = np.append(observation, BIAS_VALUE)
        model_output = model(new_observation)

        if len(model_output) > 1:
            observation, reward, done, info = env.step(np.argmax(model_output))
        else:
            observation, reward, done, info = env.step(model_output[0].numpy())
        last_reward += reward

        if render:
            env.render()

        # give a penalty if finished abrupt
        if done:
            last_reward -= 1000 - i
            break
    if render:
        print("Presentation Reward {}".format(last_reward))
    return last_reward


# evaluate models, average performance with single weight over multiple runs in environment and try with different weights
def eveluateModel(env, modelwrapper):
    tries_for_average = 5
    max_reward = -1000
    combined_reward = 0
    average = 0.0
    for weight in SHARED_WEIGHTS:
        modelwrapper.tf_model.changeSharedWeight(weight)
        average_sum = 0.0
        for _ in range(tries_for_average):

            last_reward = run_model(env, modelwrapper.tf_model, False)
            average_sum += last_reward
            # print(last_reward)

            # if last_reward > 0:
             #   observation = env.reset()

            average = average_sum / tries_for_average

            # max_reward = last_reward if (last_reward > max_reward) else max_reward
        if average > max_reward:
            max_reward = average
            modelwrapper.best_weight = weight


        combined_reward += average
    modelwrapper.max_reward = max_reward
    modelwrapper.average_reward = combined_reward / len(SHARED_WEIGHTS)
    print(modelwrapper)





def rankModels(env, model_wrapper_list, n_winner_elems):
    RANDOM_ELEMENTS = 1
    # a ratio of 80 / 20 as in the paper makes send when populations get bigger,
    # we decided for 50 / 50 in our tirals to see better results
    n_by_avgMax = int((float(n_winner_elems-RANDOM_ELEMENTS) / 10.0) * 5.0)
    n_by_avgComp = int((float(n_winner_elems-RANDOM_ELEMENTS) / 10.0) * 5.0)


    n_by_avgComp += 1 if n_by_avgMax + n_by_avgComp < n_winner_elems-RANDOM_ELEMENTS else 0

    # if the previos winners (first 2 list elements) haven't been ranked before, include them in ranking
    # otherwise skip for performance reasons
    begin = 2 if bool(model_wrapper_list[0].modelMetrik2 + model_wrapper_list[1].modelMetrik2) else 0
    for model_wrapper in model_wrapper_list[begin:]:
        eveluateModel(env, model_wrapper)

    model_wrapper_list.sort(key=lambda elem: elem.modelMetrik2, reverse=True)
    if n_by_avgMax < len(model_wrapper_list):
        winner_by_avgMax = model_wrapper_list[:n_by_avgMax]
        model_wrapper_list = model_wrapper_list[n_by_avgMax:]
    else:
        return model_wrapper_list


    model_wrapper_list.sort(key=lambda elem: elem.modelMetrik, reverse=True)
    if n_by_avgComp < len(model_wrapper_list):
        winner_by_avgComp = model_wrapper_list[:n_by_avgComp]
        model_wrapper_list = model_wrapper_list[n_by_avgComp:]

    else:
        return winner_by_avgMax + model_wrapper_list

    new_model_wrapper_list = []
    new_model_wrapper_list.append(winner_by_avgComp[0])
    new_model_wrapper_list.append(winner_by_avgMax[0])
    new_model_wrapper_list = new_model_wrapper_list + winner_by_avgComp[1:] + winner_by_avgMax[1:]
    for _ in range(n_winner_elems - len(winner_by_avgMax) - len(winner_by_avgComp)):
        if len(model_wrapper_list) > 1:
            new_model_wrapper_list.append(model_wrapper_list[np.random.randint(0, len(model_wrapper_list))])
        else:
            break
    return new_model_wrapper_list


# make changes to modelplan corresponding to the parameters of the mask
def modelEvolution(model_wrapper_list, evoultion_mask):
    # keep best models unmutated to compare against the new
    mutated_list = model_wrapper_list[0:2]

    # and mutate the rest based on the mask
    for index, pattern in evoultion_mask.items():
        print("len of list before mutation: {}".format(len(model_wrapper_list)))
        mutate_candidate = model_wrapper_list[index]
        for mutations in pattern:
            mutated = ModelWrapper()
            mutated.model_plan = deepcopy(mutate_candidate.model_plan)

            for _ in range(mutations):
                mutated.evolveModel(EVOLUTION_NAME_CHOICES[np.random.randint(0, len(EVOLUTION_NAME_CHOICES))])

            mutated.updateTFmodel()
            mutated_list.append(mutated)
    return mutated_list


# start and keep track of evolution sequence over multiple epochs
def startEvolution(env, models_wrapper_list, n_evolution_steps, n_winner_elems, evolution_mask):
    for i in range(n_evolution_steps-1):
        print("")
        print("Evolution Step : {} ".format(i+1))
        print("ranking models")

        # show winner in regards to both metriks avter every step
        models_wrapper_list = rankModels(env, models_wrapper_list, n_winner_elems)
        winner_compl = models_wrapper_list[0]
        winner_compl.tf_model.changeSharedWeight(winner_compl.best_weight)
        winner_compl = models_wrapper_list[0]
        winner_compl.tf_model.changeSharedWeight(winner_compl.best_weight)

        print("")
        print("Winner of Step {} by Complexity is : {}".format(i, id(winner_compl.tf_model)))
        print(winner_compl)
        for _ in range(3):
            run_model(env, winner_compl.tf_model, True)

        winner_max = models_wrapper_list[1]
        winner_max.tf_model.changeSharedWeight(winner_max.best_weight)
        winner_max = models_wrapper_list[1]
        winner_max.tf_model.changeSharedWeight(winner_max.best_weight)

        print("")
        print("Winner of Step {} by Maximum :".format(i))
        print(winner_max)
        for _ in range(3):
            run_model(env, winner_compl.tf_model, True)
        env.close()
        print("")
        print("model evolution")
        models_wrapper_list = modelEvolution(models_wrapper_list, evolution_mask)

    print("")
    print("last ranking")
    print("")
    models_wrapper_list = rankModels(env, models_wrapper_list, n_winner_elems)

    return models_wrapper_list



if __name__ == '__main__':
    N_INITAL_MODELS = 5
    N_EVOLUTION_STEPS = 1024
    N_WINNER = 5
    N_EVOLUTIONS_PER_STEP = 1



    tf.keras.backend.set_floatx('float64')



    # m_env = gym.make('CartPole-v0')
    m_env = CartPoleSwingUpEnv()
    #print(m_env.action_space.shape[0])
    INITIAL_MODELS = [2, 2, 3, 3, 3]
    EVOLUTION_MASK = {0: [3, 2, 2], 1: [2, 2], 2: [2], 3: [1], 4: [3]}

    wrapped_models_list = createModels(INITIAL_MODELS, len(m_env.observation_space.high), m_env.action_space.shape[0])
    wrapped_models_list = startEvolution(m_env, wrapped_models_list, N_EVOLUTION_STEPS, len(EVOLUTION_MASK), EVOLUTION_MASK)
    print("WINNER NETWORK")

    winner = wrapped_models_list[1]
    print("Final Winner is : {}".format(id(winner)))
    print(winner)
    print(winner.model_plan)




    input("enter to continue")
    run_model(m_env, winner.tf_model, True)

    m_env.close()
    print("fin")







