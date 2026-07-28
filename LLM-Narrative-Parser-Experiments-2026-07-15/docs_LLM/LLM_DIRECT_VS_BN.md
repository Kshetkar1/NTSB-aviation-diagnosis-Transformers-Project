# Can the LLM replace the Bayesian network? Direct probability test

gpt-4o-mini is asked for the SAME conditional probabilities Zhang's BN-dependent tables publish -- no network, no dataset, just its own knowledge. Error metric: order-of-magnitude error |log10(estimate / Zhang)| (0 = exact, 1 = off by 10x).

## Table 9 col: inoperative engine instruments

| Target | Zhang | Our BN | LLM direct | BN oom-err | LLM oom-err |
|---|---|---|---|---|---|
| loss of engine power | 0.95 | 0.9502 | 0.75 | 7.482e-05 | 0.1027 |
| forced landing | 0.1357 | 0.1357 | 0.6 | 0.0001209 | 0.6456 |
| ditching | 0.00437 | 0.004379 | 0.2 | 0.000865 | 1.661 |
| gear collapsed | 0.0096 | 0.00818 | 0.15 | 0.06951 | 1.194 |
| aircraft destroyed | 0.0133 | 0.01684 | 0.1 | 0.1024 | 0.8761 |
| substantial aircraft damage | 0.046 | 0.05121 | 0.5 | 0.04663 | 1.036 |
| serious injury (highest) | 0.0623 | 0.07602 | 0.3 | 0.08642 | 0.6826 |
| no injury | 0.9431 | 0.889 | 0.25 | 0.02564 | 0.5766 |

## Table 9 col: combustion liner failure

| Target | Zhang | Our BN | LLM direct | BN oom-err | LLM oom-err |
|---|---|---|---|---|---|
| loss of engine power | 0.5 | 0.5 | 0.85 | 4.59e-08 | 0.2304 |
| forced landing | 0.0714 | 0.07143 | 0.65 | 0.0001744 | 0.9592 |
| ditching | 0.0023 | 0.002304 | 0.25 | 0.000791 | 2.036 |
| gear collapsed | 0.0023 | 0.002304 | 0.15 | 0.0008033 | 1.814 |
| aircraft destroyed | 0.0023 | 1.471e-07 | 0.3 | 4.194 | 2.115 |
| substantial aircraft damage | 0.00363 | 3.943e-07 | 0.7 | 3.964 | 2.285 |
| serious injury (highest) | 0.000768 | 3.116e-07 | 0.4 | 3.392 | 2.717 |
| no injury | 0.9978 | 1 | 0.2 | 0.0009562 | 0.698 |

## Table 9 col: improper oil usage

| Target | Zhang | Our BN | LLM direct | BN oom-err | LLM oom-err |
|---|---|---|---|---|---|
| loss of engine power | 0.95 | 0.95 | 0.65 | 2.278e-09 | 0.1648 |
| forced landing | 0.1357 | 0.1357 | 0.5 | 4.603e-05 | 0.5664 |
| ditching | 0.00437 | 0.004378 | 0.2 | 0.0007902 | 1.661 |
| gear collapsed | 0.00437 | 0.004378 | 0.15 | 0.0007934 | 1.536 |
| aircraft destroyed | 0.00437 | 1.471e-07 | 0.3 | 4.473 | 1.837 |
| substantial aircraft damage | 0.00609 | 3.943e-07 | 0.4 | 4.189 | 1.817 |
| serious injury (highest) | 0.00146 | 3.116e-07 | 0.25 | 3.671 | 2.234 |
| no injury | 0.9958 | 1 | 0.1 | 0.001828 | 0.9982 |

## Table 9 col: engine instruments + improper oil

| Target | Zhang | Our BN | LLM direct | BN oom-err | LLM oom-err |
|---|---|---|---|---|---|
| loss of engine power | 0.99 | 0.9994 | 0.75 | 0.004103 | 0.1206 |
| forced landing | 0.1471 | 0.1428 | 0.6 | 0.01297 | 0.6105 |
| ditching | 0.00457 | 0.004606 | 0.3 | 0.00337 | 1.817 |
| gear collapsed | 0.00982 | 0.008407 | 0.25 | 0.06747 | 1.406 |
| aircraft destroyed | 0.0135 | 0.01684 | 0.4 | 0.09592 | 1.472 |
| substantial aircraft damage | 0.0463 | 0.05121 | 0.55 | 0.0438 | 1.075 |
| serious injury (highest) | 0.0623 | 0.07602 | 0.2 | 0.08642 | 0.5065 |
| no injury | 0.9429 | 0.889 | 0.15 | 0.02555 | 0.7984 |

## Table 9 col: loss of engine power

| Target | Zhang | Our BN | LLM direct | BN oom-err | LLM oom-err |
|---|---|---|---|---|---|
| loss of engine power | 1 | 1 | 0.9 | 0 | 0.04576 |
| forced landing | 0.1429 | 0.1449 | 0.6 | 0.00612 | 0.6231 |
| ditching | 0.00461 | 0.004675 | 0.1 | 0.006097 | 1.336 |
| gear collapsed | 0.00518 | 0.005557 | 0.2 | 0.03054 | 1.587 |
| aircraft destroyed | 0.00559 | 0.02919 | 0.3 | 0.7179 | 1.73 |
| substantial aircraft damage | 0.0166 | 0.02146 | 0.5 | 0.1116 | 1.479 |
| serious injury (highest) | 0.00822 | 0.01595 | 0.4 | 0.288 | 1.687 |
| no injury | 0.9899 | 0.9496 | 0.2 | 0.01805 | 0.6946 |

## Section 5.2 -- cumulative gear evidence, P(main gear collapsed)

| Evidence | Zhang | Our BN | LLM direct | BN oom-err | LLM oom-err |
|---|---|---|---|---|---|
| failure of the main gear strut | 0.25 | 0.25 | 0.85 | 2.571e-07 | 0.5315 |
| failure of the main gear strut AND the emergency ext | 0.682 | 0.5999 | 0.85 | 0.0557 | 0.09563 |
| failure of the main gear strut, the emergency extens | 0.777 | 0.7054 | 0.85 | 0.04199 | 0.039 |
| failure of the main gear strut, the emergency extens | 0.894 | 0.8549 | 0.85 | 0.01943 | 0.02192 |

- LLM sequence monotonically increasing (as evidence accumulates)? **YES** (Zhang and the BN both increase)

## Score

| Method | median oom-err | mean oom-err | worst | n cells |
|---|---|---|---|---|
| Our BN | 0.022 | 0.588 | 4.473 | 44 |
| LLM direct | 1.017 | 1.094 | 2.717 | 44 |

oom-err 0.30 = off by 2x, 1.00 = off by 10x, 2.00 = off by 100x.
