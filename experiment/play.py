import click
import os
import numpy as np
import h5py
import torch
import pickle
import sys

sys.path.append('/home/rjangir/software/workSpace/Overcoming-exploration-from-demos/')

from baselines import logger
from baselines.common import set_global_seeds
import config
from rollout import RolloutWorker, RolloutWorkerOriginal


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=100)
@click.option('--render', type=int, default=1)
def main(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    if params['env_name'] == 'GazeboWAMemptyEnv-v1':
        eval_params = {
            'exploit': True,
            'use_target_net': params['test_with_polyak'],
            'compute_Q': True,
            'rollout_batch_size': 1,
            #'render': bool(render),
        }

        for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
            eval_params[name] = params[name]

        madeEnv = config.cached_make_env(params['make_env'])
        evaluator = RolloutWorker(madeEnv, params['make_env'], policy, dims, logger, **eval_params)
        evaluator.seed(seed)
    else:
        eval_params = {
            'exploit': True,
            'use_target_net': params['test_with_polyak'],
            'compute_Q': True,
            'rollout_batch_size': 1,
            'render': bool(render),
        }

        for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
            eval_params[name] = params[name]

        evaluator = RolloutWorkerOriginal(params['make_env'], policy, dims, logger, **eval_params)
        evaluator.seed(seed)

    # Run evaluation.
    states, actions, rewards, lens = [], [], [], []
    success = 0
    evaluator.clear_history()
    for i in range(n_test_rollouts * 10):
        eps = evaluator.generate_rollouts()
        # eps.keys(): ['o', 'u', 'g', 'ag', 'info_is_success']
        # eps.o: (1, 51, 25)
        # eps.u: (1, 50, 4)
        # eps.g: (1, 50, 3)
        # eps.ag: (1, 51, 3)
        # eps.success: (1, 50, 1)

        if np.sum(eps["info_is_success"]) < 20.0:
            continue
        else:
            success = success + 1

        states.append(np.concatenate((eps["o"][:, :-1, :], eps["g"]), axis=2))
        actions.append(eps["u"])
        rewards.append(eps["info_is_success"])
        lens.append(eps["info_is_success"].shape[1])

        print("episode: ", i, " length: ", eps["info_is_success"].shape[1],
              " returns: ", np.sum(eps["info_is_success"]), " success: ", success)

        if success >= n_test_rollouts:
            break

    states, actions, rewards, lens = np.array(states), np.array(actions), np.array(rewards), np.array(lens)
    states, actions, rewards = np.squeeze(states, axis=1), np.squeeze(actions, axis=1), np.squeeze(rewards, axis=1)

    assert len(states) == len(actions), 'len(states) != len(actions)'
    assert len(states) == len(rewards), 'len(states) != len(rewards)'
    assert len(states) == len(lens), 'len(states) != len(lens)'

    print("=================================================================")
    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()

    if torch.cuda.is_available():
        save_dir = '/home/slee01/PycharmProjects/pytorch-a2c-ppo-acktr-gail/gail_experts/'
    else:
        save_dir = '/Users/slee01/PycharmProjects/pytorch-a2c-ppo-acktr-gail/gail_experts/'

    save_path = os.path.join(
        save_dir,
        "trajs_{}_{}.h5".format(params['env_name'].split('-')[0].lower(), "her"))

    h5f = h5py.File(save_path, 'w')

    h5f.create_dataset('obs_B_T_Do', data=states)
    h5f.create_dataset('a_B_T_Da', data=actions)
    h5f.create_dataset('r_B_T', data=rewards)
    h5f.create_dataset('len_B', data=lens)

    key_list = list(h5f.keys())

    h5f.close()

    print("env_name: ", params['env_name'])
    print("saved file keys: ", key_list)
    print("saved file name: ", save_path)

    print("expert_states: ", states.shape)
    print("expert_actions: ", actions.shape)
    print("expert_rewards: ", rewards.shape)
    print("expert_lens: ", lens.shape)
    print("=================================================================")

if __name__ == '__main__':
    main()
