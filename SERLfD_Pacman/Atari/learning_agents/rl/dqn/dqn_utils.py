import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_REG = 1e-8


def calculate_dqn_loss(model, target_model, experiences, gamma, use_double_q_update=False, reward_clip=None):
    """Return element-wise dqn loss and Q-values."""
    states, actions, rewards, next_states, dones = experiences[:5]
    if reward_clip is not None:
        rewards = torch.clamp(rewards, min=reward_clip[0], max=reward_clip[1])

    # compute current values
    if isinstance(states, tuple):
        q_values = model(*states)
    else:
        q_values = model(states)

    # According to noisynet paper,
    # it re-samples noisynet parameters on online network when using double q
    # but we don't because there is no remarkable difference in performance.
    if isinstance(next_states, tuple):
        next_q_values = model(*next_states)
    else:
        next_q_values = model(next_states)

    curr_q_value = q_values.gather(1, actions.long().unsqueeze(1))

    # compute target values
    if isinstance(next_states, tuple):
        next_target_q_values = target_model(*next_states)
    else:
        next_target_q_values = target_model(next_states)

    if use_double_q_update:
        # Double DQN
        next_q_value = next_target_q_values.gather(
            1, next_q_values.argmax(1).unsqueeze(1)
        )
    else:
        # Ordinary DQN
        next_q_value = next_target_q_values.gather(
            1, next_target_q_values.argmax(1).unsqueeze(1)
        )

    # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
    #       = r                       otherwise
    masks = 1 - dones
    target = rewards + gamma * next_q_value * masks
    target = target.to(device)

    # calculate dq loss
    dq_loss_element_wise = F.mse_loss(curr_q_value, target.detach(), reduction="none")

    return dq_loss_element_wise, q_values


def calculate_soft_dqn_loss(model, target_model, experiences, gamma,
                            alpha=0.1, use_double_q_update=False, reward_clip=None):
    """ Return element-wise soft dqn loss and Q-values. """
    states, actions, rewards, next_states, dones = experiences[:5]
    if reward_clip is not None:
        rewards = torch.clamp(rewards, min=reward_clip[0], max=reward_clip[1])

    # compute current values
    if isinstance(states, tuple):
        q_values = model(*states)
    else:
        q_values = model(states)

    if isinstance(next_states, tuple):
        next_q_values = model(*next_states)
    else:
        next_q_values = model(next_states)

    curr_q_value = q_values.gather(1, actions.long().unsqueeze(1))

    # compute target values
    if isinstance(next_states, tuple):
        next_target_q_values = target_model(*next_states)
    else:
        next_target_q_values = target_model(next_states)

    if use_double_q_update:
        # Double DQN
        next_soft_value = compute_soft_values(next_q_values,
                                              next_target_q_values, alpha=alpha)
    else:
        # Ordinary DQN
        next_soft_value = compute_soft_values(next_target_q_values,
                                              next_target_q_values, alpha=alpha)

    # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
    #       = r                       otherwise
    masks = 1 - dones
    target = rewards + gamma * next_soft_value * masks
    target = target.to(device)

    # calculate dq loss
    dq_loss_element_wise = F.mse_loss(curr_q_value, target.detach(), reduction="none")

    return dq_loss_element_wise, q_values


def compute_soft_values(q_values, q_target_values, alpha=0.1):
    # compute action probabilities using q_values
    alpha_q_values = 1 / alpha * q_values
    # for numerical stability
    action_probs = (F.log_softmax(alpha_q_values, dim=-1)).exp()

    # compute soft values using q_target_values
    q_target_values = (1 / alpha) * q_target_values
    # for numerical stability
    max_q_targets = torch.max(q_target_values, dim=-1, keepdim=True)[0]
    stable_q_targets = q_target_values - max_q_targets

    exp_q_targets = stable_q_targets.exp()
    weighted_exp_q_target_values = action_probs * exp_q_targets
    avg_exp_q_targets = torch.sum(weighted_exp_q_target_values, dim=-1, keepdim=True)
    soft_values = alpha * ((avg_exp_q_targets + LOG_REG).log() + max_q_targets)

    return soft_values

