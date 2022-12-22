import copy
import logging
import os
import time

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.multiprocessing as mp

import rlmeta.utils.hydra_utils as hydra_utils
import rlmeta.utils.random_utils as random_utils
import rlmeta.utils.remote_utils as remote_utils

from rlmeta.agents.agent import AgentFactory
from rlmeta.agents.ppo.ppo_agent import PPOAgent
from rlmeta.core.controller import Phase, Controller
from rlmeta.core.loop import LoopList, ParallelLoop
from rlmeta.core.model import ModelVersion, RemotableModelPool
from rlmeta.core.model import make_remote_model, wrap_downstream_model
from rlmeta.core.replay_buffer import ReplayBuffer, make_remote_replay_buffer
from rlmeta.core.server import Server, ServerList
from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.types import Action, TimeStep
from rlmeta.samplers import UniformSampler
from rlmeta.storage import TensorCircularBuffer
from rlmeta.utils.optimizer_utils import make_optimizer

import model_utils

from cache_env_wrapper import CacheEnvCycloneWrapperFactory
from metric_callbacks import CycloneMetricCallbacks


@hydra.main(config_path="./config", config_name="ppo_cyclone")
def main(cfg):
    if cfg.seed is not None:
        random_utils.manual_seed(cfg.seed)

    print(f"workding_dir = {os.getcwd()}")
    my_callbacks = CycloneMetricCallbacks()
    logging.info(hydra_utils.config_to_json(cfg))

    env_fac = CacheEnvCycloneWrapperFactory(
        OmegaConf.to_container(cfg.env_config))
    env = env_fac(index=0)

    train_model = model_utils.get_model(
        cfg.model_config, cfg.env_config.window_size,
        env.action_space.n).to(cfg.train_device)
    infer_model = copy.deepcopy(train_model).to(cfg.infer_device)
    infer_model.eval()
    optimizer = make_optimizer(train_model.parameters(), **cfg.optimizer)

    ctrl = Controller()
    rb = ReplayBuffer(TensorCircularBuffer(cfg.replay_buffer_size),
                      UniformSampler())

    m_server = Server(cfg.m_server_name, cfg.m_server_addr)
    r_server = Server(cfg.r_server_name, cfg.r_server_addr)
    c_server = Server(cfg.c_server_name, cfg.c_server_addr)
    m_server.add_service(RemotableModelPool(infer_model, seed=cfg.seed))
    r_server.add_service(rb)
    c_server.add_service(ctrl)
    servers = ServerList([m_server, r_server, c_server])

    a_model = wrap_downstream_model(train_model, m_server)
    t_model = make_remote_model(infer_model, m_server)
    e_model = make_remote_model(infer_model, m_server)

    a_ctrl = remote_utils.make_remote(ctrl, c_server)
    t_ctrl = remote_utils.make_remote(ctrl, c_server)
    e_ctrl = remote_utils.make_remote(ctrl, c_server)

    a_rb = make_remote_replay_buffer(rb, r_server, prefetch=cfg.prefetch)
    t_rb = make_remote_replay_buffer(rb, r_server)

    agent = PPOAgent(a_model,
                     replay_buffer=a_rb,
                     controller=a_ctrl,
                     optimizer=optimizer,
                     batch_size=cfg.batch_size,
                     learning_starts=cfg.get("learning_starts", None),
                     entropy_coeff=cfg.get("entropy_coeff", 0.01),
                     model_push_period=cfg.model_push_period)
    t_agent_fac = AgentFactory(PPOAgent, t_model, replay_buffer=t_rb)
    e_agent_fac = AgentFactory(PPOAgent, e_model, deterministic_policy=True)

    t_loop = ParallelLoop(env_fac,
                          t_agent_fac,
                          t_ctrl,
                          running_phase=Phase.TRAIN,
                          should_update=True,
                          num_rollouts=cfg.num_train_rollouts,
                          num_workers=cfg.num_train_workers,
                          seed=cfg.seed,
                          episode_callbacks=my_callbacks)
    e_loop = ParallelLoop(env_fac,
                          e_agent_fac,
                          e_ctrl,
                          running_phase=Phase.EVAL,
                          should_update=False,
                          num_rollouts=cfg.num_eval_rollouts,
                          num_workers=cfg.num_eval_workers,
                          seed=(None if cfg.seed is None else cfg.seed +
                                cfg.num_train_rollouts),
                          episode_callbacks=my_callbacks)
    loops = LoopList([t_loop, e_loop])

    servers.start()
    loops.start()
    agent.connect()

    start_time = time.perf_counter()
    for epoch in range(cfg.num_epochs):
        stats = agent.train(cfg.steps_per_epoch)
        cur_time = time.perf_counter() - start_time
        info = f"T Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Train", epoch=epoch, time=cur_time))
        time.sleep(1)

        stats = agent.eval(cfg.num_eval_episodes)
        cur_time = time.perf_counter() - start_time
        info = f"E Epoch {epoch}"
        if cfg.table_view:
            logging.info("\n\n" + stats.table(info, time=cur_time) + "\n")
        else:
            logging.info(
                stats.json(info, phase="Eval", epoch=epoch, time=cur_time))

        torch.save(train_model.state_dict(), f"ppo_agent-{epoch}.pth")
        time.sleep(1)

    loops.terminate()
    servers.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
