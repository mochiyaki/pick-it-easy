"""Minimal robot-arm imitation-learning demo using Crusoe-hosted Nemotron.

Nemotron acts as the teacher. The script records the teacher's actions in a
small 2-D simulator, then trains a linear policy to imitate those actions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass

import numpy as np
from openai import OpenAI


DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_BASE_URL = "https://api.inference.crusoecloud.com/v1"


@dataclass
class ArmSimulator:
    """A two-link planar arm with joint-delta actions."""

    theta1: float = 0.0
    theta2: float = 0.0
    link1: float = 1.0
    link2: float = 1.0
    max_step: float = 0.25

    def reset(self, rng: random.Random) -> None:
        self.theta1 = rng.uniform(-math.pi, math.pi)
        self.theta2 = rng.uniform(-math.pi, math.pi)

    def position(self) -> tuple[float, float]:
        elbow_x = self.link1 * math.cos(self.theta1)
        elbow_y = self.link1 * math.sin(self.theta1)
        return (
            elbow_x + self.link2 * math.cos(self.theta1 + self.theta2),
            elbow_y + self.link2 * math.sin(self.theta1 + self.theta2),
        )

    def features(self, target: tuple[float, float]) -> np.ndarray:
        x, y = self.position()
        error_x, error_y = target[0] - x, target[1] - y
        # These products let a linear policy represent the arm Jacobian.
        s1, c1 = math.sin(self.theta1), math.cos(self.theta1)
        s12, c12 = math.sin(self.theta1 + self.theta2), math.cos(
            self.theta1 + self.theta2
        )
        return np.array(
            [
                1.0,
                error_x,
                error_y,
                error_x * s1,
                error_x * s12,
                error_y * c1,
                error_y * c12,
            ]
        )

    def step(self, action: np.ndarray) -> None:
        clipped = np.clip(action, -self.max_step, self.max_step)
        self.theta1 = wrap_angle(self.theta1 + float(clipped[0]))
        self.theta2 = wrap_angle(self.theta2 + float(clipped[1]))

    def distance(self, target: tuple[float, float]) -> float:
        x, y = self.position()
        return math.hypot(target[0] - x, target[1] - y)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class NemotronTeacher:
    def __init__(self, model: str) -> None:
        api_key = os.getenv("CRUSOE_API_KEY")
        if not api_key:
            raise RuntimeError("Set CRUSOE_API_KEY or run with --mock.")
        base_url = os.getenv("CRUSOE_BASE_URL", DEFAULT_BASE_URL)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def act(self, arm: ArmSimulator, target: tuple[float, float]) -> np.ndarray:
        x, y = arm.position()
        prompt = f"""
Control a two-link planar robot arm. Each link is length 1.0.
Current joint angles: theta1={arm.theta1:.4f}, theta2={arm.theta2:.4f} radians.
End effector: x={x:.4f}, y={y:.4f}. Target: x={target[0]:.4f}, y={target[1]:.4f}.
Choose one delta for each joint in [-{arm.max_step}, {arm.max_step}] that moves
the end effector closer to the target. Return only JSON:
{{"dtheta1": number, "dtheta2": number}}
""".strip()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise robot-arm controller."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=80,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = response.choices[0].message.content or ""
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Nemotron did not return JSON: {text!r}")
        data = json.loads(match.group())
        return np.array([float(data["dtheta1"]), float(data["dtheta2"])])


class MockTeacher:
    """Jacobian-transpose controller used for a free, offline demonstration."""

    def act(self, arm: ArmSimulator, target: tuple[float, float]) -> np.ndarray:
        x, y = arm.position()
        error = np.array([target[0] - x, target[1] - y])
        s1, c1 = math.sin(arm.theta1), math.cos(arm.theta1)
        s12, c12 = math.sin(arm.theta1 + arm.theta2), math.cos(arm.theta1 + arm.theta2)
        jacobian = np.array(
            [[-arm.link1 * s1 - arm.link2 * s12, -arm.link2 * s12],
             [arm.link1 * c1 + arm.link2 * c12, arm.link2 * c12]]
        )
        return np.clip(0.35 * jacobian.T @ error, -arm.max_step, arm.max_step)


def random_target(rng: random.Random) -> tuple[float, float]:
    radius = rng.uniform(0.3, 1.8)
    angle = rng.uniform(-math.pi, math.pi)
    return radius * math.cos(angle), radius * math.sin(angle)


def collect_demonstrations(
    teacher: NemotronTeacher | MockTeacher,
    episodes: int,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    arm = ArmSimulator()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for episode in range(episodes):
        arm.reset(rng)
        target = random_target(rng)
        start = arm.distance(target)
        for _ in range(steps):
            observation = arm.features(target)
            action = teacher.act(arm, target)
            observations.append(observation)
            actions.append(np.clip(action, -arm.max_step, arm.max_step))
            arm.step(action)
            if arm.distance(target) < 0.08:
                break
        print(f"episode {episode + 1:02d}: distance {start:.3f} -> {arm.distance(target):.3f}")

    return np.vstack(observations), np.vstack(actions)


def train_policy(observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Fit action = observation @ weights with ordinary least squares."""
    weights, _, _, _ = np.linalg.lstsq(observations, actions, rcond=None)
    predictions = observations @ weights
    print(f"training MSE: {np.mean((predictions - actions) ** 2):.6f}")
    return weights


def evaluate(weights: np.ndarray, episodes: int, steps: int, seed: int) -> float:
    rng = random.Random(seed + 1)
    arm = ArmSimulator()
    successes = 0
    for _ in range(episodes):
        arm.reset(rng)
        target = random_target(rng)
        for _ in range(steps):
            arm.step(arm.features(target) @ weights)
        successes += arm.distance(target) < 0.35
    rate = successes / episodes
    print(f"evaluation success: {successes}/{episodes} ({rate:.0%})")
    return rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="use the offline teacher")
    parser.add_argument("--model", default=os.getenv("CRUSOE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="policy.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher = MockTeacher() if args.mock else NemotronTeacher(args.model)
    observations, actions = collect_demonstrations(
        teacher, args.episodes, args.steps, args.seed
    )
    weights = train_policy(observations, actions)
    evaluate(weights, args.eval_episodes, args.steps * 4, args.seed)
    np.savez(args.output, weights=weights)
    print(f"saved policy to {args.output}")


if __name__ == "__main__":
    main()
