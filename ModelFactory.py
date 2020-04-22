from WeightAgnosticNetwork import CustomModel
from MLPModel import MLP
import tensorflow as tf
import gym
import numpy as np

from gym_cartpole_swingup.envs import CartPoleSwingUpEnv
from copy import deepcopy
from tensorboard import main as tb


BIAS_VALUE = 1
SHARED_WEIGHTS_ = {-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2}
SHARED_WEIGHTS = [-2, -1, -0.5, 0.5, 1, 2]

EVOLUTION_NAME_CHOICES = ['add', 'insert', 'change']

EVOLUTION_NAMES = {'add': MLP.addConnection, 'insert': MLP.insertNode, 'change': MLP.changeActivation}


class ModelWrapper:
    def __init__(self):
        self.tf_model = None
        self.model_plan = None
        self.max_reward = 0
        self.average_reward = 0
        self.best_weight = 1

    def updateTFmodel(self):
        self.tf_model = CustomModel(self.model_plan, -1)

    def evolveModel(self, method):
        # method_name = EVOLUTION_NAMES[method]
        # func = getattr(self, method_name, lambda: 'Invalid')
        func = EVOLUTION_NAMES[method]
        func(self.model_plan)
        self.updateTFmodel()

    @property
    def modelMetrik2(self):
        return (self.max_reward + self.average_reward) / 2

    @property
    def modelMetrik(self):
        return self.average_reward / self.model_plan.complexity




def createModels(model_mask, n_inputs, n_outputs):
    models = []
    for model_cons in model_mask:
        wrapper = ModelWrapper()
        wrapper.model_plan = MLP(n_inputs, n_outputs, model_cons)
        wrapper.updateTFmodel()

        models.append(wrapper)
    return models


def run_model(env, model, render=False, steps=1000):
    last_reward = 0
    observation = env.reset()
    for i in range(steps):
        has_been_plus = False
        new_observation = np.append(observation, BIAS_VALUE)
        model_output = model(new_observation)

        if len(model_output) > 1:
            observation, reward, done, info = env.step(np.argmax(model_output))
        else:
            observation, reward, done, info = env.step(model_output[0].numpy())
            # reward += 1
            #if reward < 0 and not has_been_plus:
            #       reward = 0
            #else:
            #    has_been_plus = True
        last_reward += reward

        if render:
            env.render()

        if done:
            last_reward -= 1000 - i
            break
    if render:
        print(last_reward)
    return last_reward



def eveluateModel(env, modelwrapper):
    tries_for_average = 3
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
        # if last_reward < 0:
         #   last_reward = 1

        # print(last_reward)

        combined_reward += average
    modelwrapper.max_reward = max_reward
    modelwrapper.average_reward = combined_reward / len(SHARED_WEIGHTS)


"""
    print("scores")
    print(modelwrapper.tf_model)
    print(modelwrapper.max_reward)
    print(modelwrapper.average_reward)
    print(modelwrapper.modelMetrik)
    print(modelwrapper.modelMetrik2)
    print("best weight")
    print(modelwrapper.best_weight)



    print("")
"""


def rankModels(env, model_wrapper_list, n_winner_elems):
    n_by_avgMax = int((float(n_winner_elems) / 10.0) * 2.0)
    n_by_avgComp = int((float(n_winner_elems) / 10.0) * 8.0)
    n_by_avgMax = 2
    n_by_avgComp = 2

    for model_wrapper in model_wrapper_list:
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
    else:
        return model_wrapper_list

#    max_length = len(winner_by_avgMax) if len(winner_by_avgMax) > len(winner_by_avgComp) else len(winner_by_avgComp)
#    for i in range(max_length):
#        model_wrapper_list.append(winner_by_avgComp[i])
#        model_wrapper_list.append(winner_by_avgComp[i])
    model_wrapper_list.clear()
    model_wrapper_list.append(winner_by_avgComp[0])
    model_wrapper_list.append(winner_by_avgMax[0])
    model_wrapper_list = model_wrapper_list + winner_by_avgComp[1:] + winner_by_avgMax[1:]

    return model_wrapper_list


def modelEvolution(model_wrapper_list, evoultion_mask):
    print("list length {} at model evolution intern".format(len(model_wrapper_list)))
    mutated_list = []
    mutated_list = mutated_list + model_wrapper_list[0:2]

    for index, pattern in evoultion_mask.items():
        print("index {}".format(index))
        mutate_candidate = model_wrapper_list[index]
        for mutations in pattern:
            mutated = ModelWrapper()
            mutated.model_plan = deepcopy(mutate_candidate.model_plan)

            for _ in range(mutations):
                mutated.evolveModel(EVOLUTION_NAME_CHOICES[np.random.randint(0, len(EVOLUTION_NAME_CHOICES))])

            mutated.updateTFmodel()
            mutated_list.append(mutated)
    return mutated_list


"""
     multiline comment
    for mw in model_wrapper_list:
        print(mw.model_plan)
        for i in range(n_evolutions):
            print("evolve step {}".format(i))
            added = ModelWrapper()
            added.model_plan = deepcopy(mw.model_plan)
            added.evolveModel('add')
            added.updateTFmodel()
            mutated_list.append(added)

            inserted = ModelWrapper()
            inserted.model_plan = deepcopy(mw.model_plan)
            inserted.evolveModel('insert')
            inserted.updateTFmodel()
            mutated_list.append(inserted)

            changed = ModelWrapper()
            changed.model_plan = deepcopy(mw.model_plan)
            changed.evolveModel('change')
            changed.updateTFmodel()
            mutated_list.append(changed)
"""


def startEvolution(env, models_wrapper_list, n_evolution_steps, n_winner_elems, evolution_mask):
    print("initial length {}".format(len(models_wrapper_list)))

    for i in range(n_evolution_steps-1):
        print("")
        print("rank models")
        print("")
        models_wrapper_list = rankModels(env, models_wrapper_list, n_winner_elems)
        winner = models_wrapper_list[0]
        winner.tf_model.changeSharedWeight(winner.best_weight)
    
        print("")
        print("list length {} at {} after rank".format(len(models_wrapper_list), i))

        winner = models_wrapper_list[0]
        winner.tf_model.changeSharedWeight(winner.best_weight)
        print("winner score")
        print(winner.tf_model)
        print(winner.max_reward)
        print(winner.average_reward)
        print(winner.modelMetrik)
        print(winner.modelMetrik2)
        print("best weight")
        print(winner.best_weight)
        for _ in range(3):
            reward = run_model(env, winner.tf_model, True)
            print("winner reward {}".format(reward))
        env.close()
        print("")
        print("model evolution")
        print("")
        models_wrapper_list = modelEvolution(models_wrapper_list, evolution_mask)
        print("list length {} at {} after evolution".format(len(models_wrapper_list), i))



    print("")
    print("last ranking")
    print("")
    models_wrapper_list = rankModels(env, models_wrapper_list, n_winner_elems)

    return models_wrapper_list



if __name__ == '__main__':
    N_INITAL_MODELS = 5
    N_EVOLUTION_STEPS = 50
    N_WINNER = 5
    N_EVOLUTIONS_PER_STEP = 1



    tf.keras.backend.set_floatx('float64')



    # m_env = gym.make('CartPole-v0')
    m_env = CartPoleSwingUpEnv()
    #print(m_env.action_space.shape[0])
    INITIAL_MODELS = [2, 2, 3, 3, 3]
    EVOLUTION_MASK = {0: [2, 2], 1: [2, 2], 2: [1], 3: [1]}

    wrapped_models_list = createModels(INITIAL_MODELS, len(m_env.observation_space.high), m_env.action_space.shape[0])
    wrapped_models_list = startEvolution(m_env, wrapped_models_list, N_EVOLUTION_STEPS, len(EVOLUTION_MASK), EVOLUTION_MASK)
    print("WINNER NETWORK")

    winner = wrapped_models_list[0]

    print(winner.model_plan)
    winner.tf_model.changeSharedWeight(winner.best_weight)
    input("enter to continue")
    run_model(m_env, winner.tf_model, True)

    m_env.close()
    print("fin")







