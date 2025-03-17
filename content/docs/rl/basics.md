---
title: Basics
weight: 1
bookToc: true
---

# Reinforcement learning basics

{{< button href="https://www.dropbox.com/s/4pdr6y60t0r9mm2/2017_07_29_anthill_deep_reinforcement_learning_tutorial.pdf?dl=0" >}}slides{{< /button >}}

> Reinforcement Learning (RL) is a natural computational paradigm for agents learning from interaction to achieve a goal. Deep learning (DL) provides a powerful general-purpose representation learning framework. A combination of these two has recently emerged as a strong contender for artificial general intelligence. This tutorial will provide a gentle exposition of RL concepts and DL based RL with a focus on policy gradients.

{{< katex >}}{{< /katex >}}

## The agent-environment interaction

The reinforcement learning (RL) framework is an abstraction of the problem of **goal directed learning from interaction**.

- The learner and the decision maker is called the **agent**.
- The thing it interacts with (everything outside the agent) is called the **environment**.

The agent and the environment interact continually. In the RL framework any problem of learning goal-directed behavior is abstracted to three signals passing back and forth between the agent and the environment.
1. one signal to represent the choices made by the agent (the **actions**)
2. one signal to represent the basis on which the choices are made (the **states**)
3. one signal to define the agents goal  (**rewards**)

In general, actions can be any decisions we want to learn how to make, and the states can be anything we can know that might be useful in making them. More formally at each time step {{< katex >}}t{{< /katex >}}
- The agent receives a representation of the environments **state**, {{< katex >}}S_t \in \mathcal{S}{{< /katex >}}, where {{< katex >}}\mathcal{S}{{< /katex >}} is the set of possible states.
- On the basis of {{< katex >}}S_t{{< /katex >}} the agent selects an **action** {{< katex >}}A_t \in \mathcal{A}(S_t){{< /katex >}}, where {{< katex >}}\mathcal{A}(S_t){{< /katex >}} the set of actions available in state {{< katex >}}S_t{{< /katex >}}.
- *One time step later*, as a result of the action {{< katex >}}A_t{{< /katex >}} the agent receives a scalar **reward**,  {{< katex >}}R_{t+1} \in \mathcal{R} \subset \mathbb{R}{{< /katex >}}.
- The agent then observes a new state {{< katex >}}S_{t+1}{{< /katex >}}.

<img src="/img/rl.jpg"  width="600"/>

## Policy

At each time step, the agent essentially has to implement a mapping (called the agents **policy**) from states to actions.

The agents (stochastic) policy  is denoted by {{< katex >}}\pi_t{{< /katex >}}, where

{{< katex display=true >}}
\pi_t(a|s) = \mathbb{P}[A_t=a|S_t=s].
{{< /katex >}}

## Markov Decision Processes

In the most general case the environment response may depend on everything that has happened earlier.
{{< katex display=true >}}
Pr\left\{S_{t+1}=s^{'},R_{t+1}=r|S_0,A_0,R_1,...,S_{t-1},A_{t-1},R_t,S_t,A_t,\right\}.
{{< /katex >}}

If the state signal has Markov property then the response depends only on the state and action representations at {{< katex >}}t{{< /katex >}}
{{< katex display=true >}}
Pr\left\{S_{t+1}=s^{'},R_{t+1}=r|S_t,A_t,\right\}.
{{< /katex >}}

A RL task that satisfies the **Markov property** is called a Markov Decision Process (MDP).


## Goals and rewards

The agents goal is to maximize the total amount of cumulative reward it receives **over the long run** (and **not** the immediate reward). If we want the agent to achieve some goal, we must specify the rewards to in in such a way that in maximizing them the agent will also achieve our goals.

> **Episodic tasks** have a natural notion of the final time step. The agent-environment interaction naturally breaks into sub sequences, which are called **episodes**, such as plays of a game, trips through a maze, etc. Each episode ends in a special state called the *terminal state*, followed by a reset to the standard starting state.

> For  **continuing tasks** the agent-environment interaction goes on continually without limit.

Let {{< katex >}}R\_{t+1},R\_{t+2},R\_{t+3},...{{< /katex >}} be the sequence of rewards received after time step {{< katex >}}t{{< /katex >}}.

The **return** {{< katex >}}G_t{{< /katex >}} is a defined as some specific function (for example, the sum) of the reward sequence. For episodic tasks
{{< katex display=true >}}
G_t = R_{t+1}+R_{t+2}+R_{t+3}+...+R_{T},
{{< /katex >}}
where {{< katex >}}T{{< /katex >}} is the final time step.

For continuing tasks we need an additional concept of **discounting**. The **discounted return** is given by
{{< katex display=true >}}
G_t = R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+... = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1} = R_{t+1}+\gamma G_{t+1},
{{< /katex >}}
where {{< katex >}}0 \leq \gamma \leq 1{{< /katex >}} is a parameter called the **discount rate**.

The discount rate determines the present value of future rewards. A reward received {{< katex >}}k{{< /katex >}} time steps in the future is worth only {{< katex >}}\gamma^{k-1}{{< /katex >}} times what it would have been worth if it were received immediately.

The discount rate determines the present value of future rewards.

As {{< katex >}}\gamma{{< /katex >}} approaches 1 the agent becomes more farsighted.

As {{< katex >}}\gamma{{< /katex >}} approaches 0 the agent becomes more myopic.

**The agents goal is to choose the actions to maximize the expected discounted return**.

## Value Functions

A value function is a prediction of future reward. Recall that, the agents (stochastic) policy  is denoted by {{< katex >}}\pi{{< /katex >}}, where
{{< katex display=true >}}
\pi(a|s) = \mathbb{P}[A_t=a|S_t=s].
{{< /katex >}}

The **value of state** {{< katex >}}s{{< /katex >}} under a policy {{< katex >}}\pi{{< /katex >}} is the expected return when starting in {{< katex >}}s{{< /katex >}}  and following the policy {{< katex >}}\pi{{< /katex >}} thereafter.
{{< katex display=true >}}
v_{\pi}(s) = \mathbb{E}_{\pi}[G_t|S_t=s] = \mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}|S_t=s].
{{< /katex >}}
{{< katex >}}v_{\pi}{{< /katex >}} is called the **state-value function** for policy {{< katex >}}\pi{{< /katex >}}.

*How much reward will I get from state {{< katex >}}s{{< /katex >}} under policy {{< katex >}}\pi{{< /katex >}}?*

The **value of action** {{< katex >}}a{{< /katex >}} in state {{< katex >}}s{{< /katex >}} under a policy {{< katex >}}\pi{{< /katex >}} is the expected return starting in {{< katex >}}s{{< /katex >}}, taking the action {{< katex >}}a{{< /katex >}}, and thereafter following policy {{< katex >}}\pi{{< /katex >}}.

{{< katex display=true >}}
q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t|S_t=s,A_t=a]
=\mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}|S_t=s,A_t=a].
{{< /katex >}}

{{< katex >}}q_{\pi}{{< /katex >}} is called the **action-value function** or the **Q-value function** for policy {{< katex >}}\pi{{< /katex >}}.

*How much reward will I get from action {{< katex >}}a{{< /katex >}} in state {{< katex >}}s{{< /katex >}} under policy {{< katex >}}\pi{{< /katex >}}?*

Value functions decompose into a **Bellman equation** which specifies the relation between the value of {{< katex >}}s{{< /katex >}} and the value of its possible successor states.

{{< katex display=true >}}
v_{\pi}(s) =  \mathbb{E}[r+\gamma v_{\pi}(s^{'})]
{{< /katex >}}

## Optimal Value Functions

*The agents goal is to find a policy to maximize the expected discounted return*.

**Why are we talking about value functions ?**

Value functions define a partial ordering over polices.

{{< katex display=true >}}
\pi \geq \pi{'} \text{  if and only if  } v_{\pi}(s) \geq v_{\pi}(s^{'}) \text{  for all } s \in \mathcal{S}
{{< /katex >}}

The optimal policy is the one which has the maximum state-value function.

**optimal state-value function**

{{< katex display=true >}}
v_{*}(s) = \max_{\pi} v_{\pi}(s) \text{ for all } s \in \mathcal{S}
{{< /katex >}}

Bellman's optimality equation

{{< katex display=true >}}
v_{*}(s) = \max_{a \in \mathcal{A}(s)} \sum_{s^{'},r} p(s^{'},r|s,a)[r+\gamma v_{*}(s^{'}) ]
{{< /katex >}}

**optimal action-value function**

{{< katex display=true >}}
q_{*}(s,a) = \max_{\pi} q_{\pi}(s,a) \text{ for all } s \in \mathcal{S} \text{ and } a \in \mathcal{A}
{{< /katex >}}

Bellman's optimality equation

{{< katex display=true >}}
q_{*}(s,a) = \sum_{s^{'},r} p(s^{'},r|s,a)[r+\gamma \max_{a^{'}} q_{*}(s^{'},a^{'}) ]
{{< /katex >}}

The value of the start state must equal the discounted value of the expected next state plus the reward expected along the way.

## Policy gradient methods

The goal is to learn a **parametrized policy** that can select actions without consulting a value function. Note that a value function will still be used to **learn** the policy parameter, but is not required for action selection.

Let {{< katex >}}\theta \in \mathbb{R}^{d}{{< /katex >}} represent the policy's parameter vector. The parameterized policy is written as

{{< katex display=true >}}
\pi(a|s,\theta) = \text{Pr}[A_t=a|S_t=s,\theta_t=\theta]
{{< /katex >}}

This is the probability that action {{< katex >}}a{{< /katex >}} is taken at time {{< katex >}}t{{< /katex >}} given that the agent is in state {{< katex >}}s{{< /katex >}} at time {{< katex >}}t{{< /katex >}} with parameter {{< katex >}}\theta{{< /katex >}}.

We will estimate the policy parameter  {{< katex >}}\theta{{< /katex >}} to maximize a performance measure {{< katex >}}J(\theta){{< /katex >}}.

{{< katex display=true >}}
\widehat{\theta} = \arg \max_{\theta} J(\theta)
{{< /katex >}}

As usual we will use stochastic gradient *ascent*

{{< katex display=true >}}
\theta_{t+1} = \theta_(t) + \alpha \widehat{\nabla J(\theta_t)},
{{< /katex >}}
where {{< katex >}}\widehat{\nabla J(\theta_t)}{{< /katex >}} is a stochastic estimate of the gradient whose expectation approximates the true gradient.

For the episodic case the performance is defined as the value of the start state under the parameterized policy.

{{< katex display=true >}}
J(\theta) = v_{\pi_{\theta}}(s_0)
{{< /katex >}}

Recall,  the **value of a state** {{< katex >}}s{{< /katex >}} under a policy {{< katex >}}\pi{{< /katex >}} is the expected return when starting in {{< katex >}}s{{< /katex >}}  and following the policy {{< katex >}}\pi{{< /katex >}} thereafter.

{{< katex display=true >}}
v_{\pi_\theta}(s_0) = \mathbb{E}_{\pi_\theta}[G_t|S_t=s_0]
= \mathbb{E}_{\pi_\theta}[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}|S_t=s_0]
{{< /katex >}}

> **Log-Derivative trick**
If {{< katex >}}x \sim p\_{\theta}(.){{< /katex >}} then
{{< katex display=true >}}
\nabla_\theta \mathbb{E}[f(x)] = \nabla_\theta  \int p_{\theta}(x) f(x) dx = \int \frac{p_{\theta}(x)}{p_{\theta}(x)} \nabla_\theta  p_{\theta}(x) f(x) dx
{{< /katex >}}
{{< katex display=true >}}
\nabla_\theta \mathbb{E}[f(x)] = \int p_{\theta}(x) \nabla_\theta \log p_{\theta}(x) f(x) dx = \mathbb{E}[f(x)\nabla_\theta \log  p_{\theta}(x)]
{{< /katex >}}

We also need the compute the gradient of the log probability of an episode

**Gradient of the log probability of an episode**

Let {{< katex >}}\tau{{< /katex >}} be an episode of length {{< katex >}}T{{< /katex >}} defined as
{{< katex display=true >}}
\tau=(s_0,a_0,r_1,s_1,a_1,r_2,....,a_{T-1},r_T,s_T).
{{< /katex >}}
Then
{{< katex display=true >}}
\nabla_{\theta} \log p_{\theta}(\tau) = \nabla_{\theta} \log \left(\mu(s_0) \prod_{t=0}^{T-1} \pi_{\theta}(a_t|s_t)\text{Pr}(s_{t+1}|s_t,a_t) \right)
{{< /katex >}}
{{< katex display=true >}}
\nabla_{\theta} \log p_{\theta}(\tau) = \nabla_{\theta} \left[\log \mu(s_0) + \sum_{t=0}^{T-1} ( \log \pi_{\theta}(a_t|s_t)+ \log \text{Pr}(s_{t+1}|s_t,a_t) ) \right]
{{< /katex >}}
{{< katex display=true >}}
\nabla_{\theta} \log p_{\theta}(\tau) = \nabla_{\theta}  \sum_{t=0}^{T-1} \log \pi_{\theta}(a_t|s_t)
{{< /katex >}}

> Observe that when taking gradients, the state dynamics disappear!

Using the above two tricks

{{< katex display=true >}}
\nabla_{\theta} v_{\pi_\theta}(s_0) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ G_{\tau}\nabla_{\theta}   \sum_{t=0}^{T-1} \log \pi_{\theta}(a_t|s_t)\right]
{{< /katex >}}


## References

- http://karpathy.github.io/2016/05/31/rl/
- https://www.davidsilver.uk/teaching/
- http://incompleteideas.net/sutton/book/the-book-2nd.html
- http://rll.berkeley.edu/deeprlcourse/
- https://gym.openai.com/
- https://deepmind.com/blog/deep-reinforcement-learning/
- http://www.scholarpedia.org/article/Policy_gradient_methods#Assumptions_and_Notation
- https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/
- http://www.1-4-5.net/~dmm/ml/log_derivative_trick.pdf
