import gym
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    # 比较函数，返回1, 0, -1
    return int((a > b)) - int((a < b))

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    # 抽牌
    return np_random.choice(deck)


def draw_hand(np_random):
    # 抽两张牌做一手牌
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  
    # Does this hand have a usable ace?
    # 检查手中是否有可用的Ace以及综合不超过21
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
            return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust爆掉?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (1998).
    https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html
    """
    def __init__(self, natural=False):
        # 定义动作空间 (0停止 1继续要牌)
        self.action_space = spaces.Discrete(2)
        
        # 定义观察空间 (玩家点数、 庄家明牌、 是否有可用Ace)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self._seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self._reset()        # Number of 
        self.nA = 2

    def reset(self):
        # 重置环境并返回初始观察
        return self._reset()

    def step(self, action):
        # 执行动作并返回新的观察、奖励、是否结束、额外信息
        return self._step(action)

    def _seed(self, seed=None):
        # 设置随机数种子
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1 # 爆掉就-1
            else:
                done = False
                reward = 0 # 没爆掉，奖励为0继续游戏
                
        else:  # stick: play out the dealers hand, and score
            done = True
            # 庄家一直抽牌到 >= 17
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            # 比较双方点数
            reward = cmp(score(self.player), score(self.dealer))
            
            # 自然获胜，得1.5在
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # observation 返回当前观察状态
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def _reset(self):
        # 重置，各抽两张牌
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)

        # Auto-draw another card if the score is less than 12
        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.np_random))

        return self._get_obs()
