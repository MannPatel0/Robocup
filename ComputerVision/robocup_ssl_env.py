# import pygame
# import numpy as np

# # Constants based on the image provided
# FIELD_WIDTH = 13.4  # meters
# FIELD_HEIGHT = 10.4  # meters
# GOAL_WIDTH = 1.8  # meters
# GOAL_HEIGHT = 1.8  # meters
# GOAL_DEPTH = 0.7  # meters
# SCALE = 50  # pixels per meter
# FPS = 60

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# BLUE = (0, 0, 255)
# GREEN = (0, 255, 0)
# RED = (255, 0, 0)
# ORANGE = (255, 165, 0)

# class RoboCupSSLEnv:
#     def __init__(self):
#         pygame.init()
#         self.screen = pygame.display.set_mode((int(FIELD_WIDTH * SCALE), int(FIELD_HEIGHT * SCALE)))
#         pygame.display.set_caption("RoboCup SSL Environment")
#         self.clock = pygame.time.Clock()

#         self.total_reward = 0
#         self._reset_positions()

#     def _reset_positions(self):
#         self.robot_pos = np.array([6.7, 5.2])
#         self.robot_angle = 0
#         self.ball_pos = np.array([6.7, 3.2])
#         self.ball_in_possession = False

#     def reset(self):
#         self.total_reward = 0
#         self._reset_positions()
#         return self._get_obs()

#     def _get_obs(self):
#         return np.array([
#             self.robot_pos[0], self.robot_pos[1], self.robot_angle,
#             self.ball_pos[0], self.ball_pos[1], int(self.ball_in_possession)
#         ])

#     def step(self, action):
#         if action == 0:  # Turn left
#             self.robot_angle -= np.pi / 18  # Turn 10 degrees
#         elif action == 1:  # Turn right
#             self.robot_angle += np.pi / 18  # Turn 10 degrees
#         elif action == 2:  # Move forward
#             self.robot_pos[0] += 0.1 * np.cos(self.robot_angle)
#             self.robot_pos[1] += 0.1 * np.sin(self.robot_angle)
#         elif action == 3:  # Move backward
#             self.robot_pos[0] -= 0.1 * np.cos(self.robot_angle)
#             self.robot_pos[1] -= 0.1 * np.sin(self.robot_angle)
#         elif action == 4:  # Kick
#             if self.ball_in_possession:
#                 self.ball_pos = self.robot_pos + 2 * np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
#                 self.ball_in_possession = False

#         # Ball possession
#         if not self.ball_in_possession and np.linalg.norm(self.robot_pos - self.ball_pos) < 0.2:
#             self.ball_in_possession = True

#         # Move ball with robot if in possession
#         if self.ball_in_possession:
#             self.ball_pos = self.robot_pos + np.array([0.2 * np.cos(self.robot_angle), 0.2 * np.sin(self.robot_angle)])

#         # Collision with field boundaries
#         self.robot_pos = np.clip(self.robot_pos, [0, 0], [FIELD_WIDTH, FIELD_HEIGHT])
#         self.ball_pos = np.clip(self.ball_pos, [0, 0], [FIELD_WIDTH, FIELD_HEIGHT])

#         # Check for goal on the right side
#         reward = 0
#         done = False
#         if self.ball_pos[0] >= FIELD_WIDTH and (FIELD_HEIGHT / 2 - GOAL_HEIGHT / 2) <= self.ball_pos[1] <= (FIELD_HEIGHT / 2 + GOAL_HEIGHT / 2):
#             reward += 1  # Scored a goal
#             self.total_reward += reward
#             print("---> Goal! Total Reward:", self.total_reward)
#             self._reset_positions()  # Reset player and ball positions

#         return self._get_obs(), reward, done, {}

#     def handle_keys(self):
#         keys = pygame.key.get_pressed()
#         if keys[pygame.K_LEFT]:
#             return 0  # Turn left
#         elif keys[pygame.K_RIGHT]:
#             return 1  # Turn right
#         elif keys[pygame.K_UP]:
#             return 2  # Move forward
#         elif keys[pygame.K_DOWN]:
#             return 3  # Move backward
#         elif keys[pygame.K_SPACE]:
#             return 4  # Kick
#         return -1  # No action

#     def render(self):
#         self.screen.fill(BLACK)  # Clear screen

#         # Draw field
#         pygame.draw.rect(self.screen, GREEN, pygame.Rect(0, 0, FIELD_WIDTH * SCALE, FIELD_HEIGHT * SCALE))

#         # Draw center line
#         pygame.draw.line(self.screen, WHITE, (FIELD_WIDTH * SCALE / 2, 0), (FIELD_WIDTH * SCALE / 2, FIELD_HEIGHT * SCALE), 2)

#         # Draw center circle
#         pygame.draw.circle(self.screen, WHITE, (int(FIELD_WIDTH * SCALE / 2), int(FIELD_HEIGHT * SCALE / 2)), int(1.0 * SCALE), 2)

#         # Draw goals
#         # Left goal
#         pygame.draw.rect(self.screen, WHITE, pygame.Rect(0, (FIELD_HEIGHT / 2 - GOAL_HEIGHT / 2) * SCALE, GOAL_DEPTH * SCALE, GOAL_HEIGHT * SCALE), 2)
#         pygame.draw.rect(self.screen, WHITE, pygame.Rect(0, (FIELD_HEIGHT / 2 - GOAL_HEIGHT / 2) * SCALE, GOAL_WIDTH * SCALE, GOAL_HEIGHT * SCALE), 2)

#         # Right goal
#         pygame.draw.rect(self.screen, RED, pygame.Rect((FIELD_WIDTH - GOAL_DEPTH) * SCALE, (FIELD_HEIGHT / 2 - GOAL_HEIGHT / 2) * SCALE, GOAL_DEPTH * SCALE, GOAL_HEIGHT * SCALE), 2)
#         pygame.draw.rect(self.screen, RED, pygame.Rect((FIELD_WIDTH - GOAL_WIDTH) * SCALE, (FIELD_HEIGHT / 2 - GOAL_HEIGHT / 2) * SCALE, GOAL_WIDTH * SCALE, GOAL_HEIGHT * SCALE), 2)

#         # Draw robot
#         robot_center = (int(self.robot_pos[0] * SCALE), int(self.robot_pos[1] * SCALE))
#         pygame.draw.circle(self.screen, BLUE, robot_center, 10)

#         # Draw direction arrow
#         robot_arrow_end = (robot_center[0] + int(20 * np.cos(self.robot_angle)), robot_center[1] + int(20 * np.sin(self.robot_angle)))
#         pygame.draw.line(self.screen, BLUE, robot_center, robot_arrow_end, 3)

#         # Draw ball
#         ball_center = (int(self.ball_pos[0] * SCALE), int(self.ball_pos[1] * SCALE))
#         pygame.draw.circle(self.screen, ORANGE, ball_center, 8)

#         pygame.display.flip()
#         self.clock.tick(FPS)

#     def close(self):
#         pygame.quit()

# # Usage
# env = RoboCupSSLEnv()
# obs = env.reset()
# done = False

# while not done:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             done = True

#     action = env.handle_keys()
#     if action != -1:
#         obs, reward, done, info = env.step(action)
#     env.render()

#     if env.total_reward >= 25:
#         done = True

# env.close()
import gym
from gym import spaces
import pygame
import numpy as np

# Constants based on the image provided
FIELD_WIDTH = 13.4  # meters
FIELD_HEIGHT = 10.4  # meters
GOAL_WIDTH = 1.8  # meters
GOAL_HEIGHT = 1.8  # meters
GOAL_DEPTH = 0.7  # meters
SCALE = 50  # pixels per meter
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)

class RoboCupSSLEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RoboCupSSLEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(5)  # 5 possible actions
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, 0, 0, 0]),
            high=np.array([FIELD_WIDTH, FIELD_HEIGHT, np.pi, FIELD_WIDTH, FIELD_HEIGHT, 1]),
            dtype=np.float32
        )

        self.total_reward = 0
        self._reset_positions()

    def _reset_positions(self):
        self.robot_pos = np.array([6.7, 5.2])
        self.robot_angle = 0
        self.ball_pos = np.array([6.7, 3.2])
        self.ball_in_possession = False

    def reset(self):
        self.total_reward = 0
        self._reset_positions()
        return self._get_obs()

    def _get_obs(self):
        return np.array([
            self.robot_pos[0], self.robot_pos[1], self.robot_angle,
            self.ball_pos[0], self.ball_pos[1], int(self.ball_in_possession)
        ])

    def step(self, action):
        if action == 0:  # Turn left
            self.robot_angle -= np.pi / 18  # Turn 10 degrees
        elif action == 1:  # Turn right
            self.robot_angle += np.pi / 18  # Turn 10 degrees
        elif action == 2:  # Move forward
            self.robot_pos[0] += 0.1 * np.cos(self.robot_angle)
            self.robot_pos[1] += 0.1 * np.sin(self.robot_angle)
        elif action == 3:  # Move backward
            self.robot_pos[0] -= 0.1 * np.cos(self.robot_angle)
            self.robot_pos[1] -= 0.1 * np.sin(self.robot_angle)
        elif action == 4:  # Kick
            if self.ball_in_possession:
                self.ball_pos = self.robot_pos + 2 * np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
                self.ball_in_possession = False

        # Ball possession
        if not self.ball_in_possession and np.linalg.norm(self.robot_pos - self.ball_pos) < 0.2:
            self.ball_in_possession = True

        # Move ball with robot if in possession
        if self.ball_in_possession:
            self.ball_pos = self.robot_pos + np.array([0.2 * np.cos(self.robot_angle), 0.2 * np.sin(self.robot_angle)])

        # Collision with field boundaries
        self.robot_pos = np.clip(self.robot_pos, [0, 0], [FIELD_WIDTH, FIELD_HEIGHT])
        self.ball_pos = np.clip(self.ball_pos, [0, 0], [FIELD_WIDTH, FIELD_HEIGHT])

        # Check for goal on the right side
        reward = 0
        done = False
        if self.ball_pos[0] >= FIELD_WIDTH and (FIELD_HEIGHT / 2 - GOAL_HEIGHT / 2) <= self.ball_pos[1] <= (FIELD_HEIGHT / 2 + GOAL_HEIGHT / 2):
            reward += 1  # Scored a goal
            self.total_reward += reward
            print("---> Goal! Total Reward:", self.total_reward)
            self._reset_positions()  # Reset player and ball positions

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((int(FIELD_WIDTH * SCALE), int(FIELD_HEIGHT * SCALE)))
            pygame.display.set_caption("RoboCup SSL Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill(BLACK)  # Clear screen

        # Draw field
        pygame.draw.rect(self.screen, GREEN, pygame.Rect(0, 0, FIELD_WIDTH * SCALE, FIELD_HEIGHT * SCALE))

        # Draw center line
        pygame.draw.line(self.screen, WHITE, (FIELD_WIDTH * SCALE / 2, 0), (FIELD_WIDTH * SCALE / 2, FIELD_HEIGHT * SCALE), 2)

        # Draw center circle
        pygame.draw.circle(self.screen, WHITE, (int(FIELD_WIDTH * SCALE / 2), int(FIELD_HEIGHT * SCALE / 2)), int(1.0 * SCALE), 2)

        # Draw goals
        # Left goal
        pygame.draw.rect(self.screen, WHITE, pygame.Rect(0, (FIELD_HEIGHT / 2 - GOAL_HEIGHT / 2) * SCALE, GOAL_DEPTH * SCALE, GOAL_HEIGHT * SCALE), 2)
        pygame.draw.rect(self.screen, WHITE, pygame.Rect(0, (FIELD_HEIGHT / 2 - GOAL_HEIGHT / 2) * SCALE, GOAL_WIDTH * SCALE, GOAL_HEIGHT * SCALE), 2)

        # Right goal
        pygame.draw.rect(self.screen, RED, pygame.Rect((FIELD_WIDTH - GOAL_DEPTH) * SCALE, (FIELD_HEIGHT / 2 - GOAL_HEIGHT / 2) * SCALE, GOAL_DEPTH * SCALE, GOAL_HEIGHT * SCALE), 2)
        pygame.draw.rect(self.screen, RED, pygame.Rect((FIELD_WIDTH - GOAL_WIDTH) * SCALE, (FIELD_HEIGHT / 2 - GOAL_HEIGHT / 2) * SCALE, GOAL_WIDTH * SCALE, GOAL_HEIGHT * SCALE), 2)

        # Draw robot
        robot_center = (int(self.robot_pos[0] * SCALE), int(self.robot_pos[1] * SCALE))
        pygame.draw.circle(self.screen, BLUE, robot_center, 10)

        # Draw direction arrow
        robot_arrow_end = (robot_center[0] + int(20 * np.cos(self.robot_angle)), robot_center[1] + int(20 * np.sin(self.robot_angle)))
        pygame.draw.line(self.screen, GREEN, robot_center, robot_arrow_end, 3)

        # Draw ball
        ball_center = (int(self.ball_pos[0] * SCALE), int(self.ball_pos[1] * SCALE))
        pygame.draw.circle(self.screen, ORANGE, ball_center, 8)

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()
