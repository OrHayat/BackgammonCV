import pyglet
from pyglet import gl
import numpy as np
from collections import namedtuple
from gym_backgammon.envs.backgammon import WHITE, BLACK, NUM_POINTS,assert_board
import os
import platform

Coords = namedtuple('Coords', ['x', 'y'])

SCALING = 2 if platform.system() == 'Darwin' else 1


def clamp(x, l, u):
    return max(l, min(u, x))


class Viewer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.window = pyglet.window.Window(width=width, height=height, display=None)
        self.window.set_visible(False)
        pyglet.resource.path = [os.path.dirname(__file__) + '/resources']
        pyglet.resource.reindex()

        empty_board_image = pyglet.resource.image("board.png")
        empty_board_image.width = width
        empty_board_image.height = height

        self.empty_board_image = empty_board_image

        # CHECKERS
        self.checker_diameter = self.width / 15  # 40

        self.checkers = {
            WHITE: {i: pyglet.resource.image("white_{}.png".format(i)) for i in range(1, 16)},
            BLACK: {i: pyglet.resource.image("black_{}.png".format(i)) for i in range(1, 16)}
        }

        coords = {}
        shifts = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]

        for i in range(NUM_POINTS):
            index = 23 - i if 11 < i <= 23 else i

            if i < 12:
                coords[i] = Coords(x=width - (shifts[index] * self.checker_diameter), y=self.checker_diameter)
            else:
                coords[i] = Coords(x=width - (shifts[index] * self.checker_diameter), y=height - self.checker_diameter)

        coords['bar_{}'.format(WHITE)] = Coords(x=width - (8 * self.checker_diameter), y=height - self.checker_diameter)
        coords['bar_{}'.format(BLACK)] = Coords(x=width - (8 * self.checker_diameter), y=self.checker_diameter)

        coords['off_{}'.format(WHITE)] = Coords(x=width - self.checker_diameter, y=self.checker_diameter)
        coords['off_{}'.format(BLACK)] = Coords(x=width - self.checker_diameter, y=height - self.checker_diameter)

        self.points_coord = coords

        # remove
        self.counter = 0

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def render_action(self, pos, board, bar, off, player, batch, group, color, is_target):

        if pos == -1:  # white off
            # white_off_bot
            # if player != WHITE:
                # assert_board(action="NOT AVILABLE",board=board,bar=bar,off=off)
            assert player == WHITE, "player should be white({}) on this pos but it is {}".format(WHITE,player)
            checkers = off[player] + (0 if is_target else -1)
            checkers = clamp(x=checkers, l=0, u=4)
            c = self.points_coord['off_{}'.format(player)]
            return pyglet.shapes.Circle(c.x + self.checker_diameter // 2,
                                        c.y + self.checker_diameter // 2 + self.checker_diameter * checkers,
                                        self.checker_diameter // 2, color=color, batch=batch,
                                        group=group)

        if pos == NUM_POINTS:  # black off
            # black_off_top
            assert player == BLACK, "player should be black on this pos"
            checkers = off[player] + (0 if is_target else -1)
            checkers = clamp(x=checkers, l=0, u=4)
            c = self.points_coord['off_{}'.format(player)]
            return pyglet.shapes.Circle(c.x + self.checker_diameter // 2,
                                        c.y - self.checker_diameter // 2 - self.checker_diameter * checkers,
                                        self.checker_diameter // 2, color=color, batch=batch,
                                        group=group)

        if pos == 'bar':
            checkers = bar[player] + (0 if is_target else -1)
            checkers = clamp(x=checkers, l=1, u=4)
            c = self.points_coord['bar_{}'.format(player)]
            if player == BLACK:
                # black_bar_top
                return pyglet.shapes.Circle(c.x + self.checker_diameter // 2,
                                            c.y - self.checker_diameter // 2 - self.checker_diameter * checkers,
                                            self.checker_diameter // 2, color=color, batch=batch,
                                            group=group)
            elif player == WHITE:
                # bottom_bar_white
                return pyglet.shapes.Circle(c.x + self.checker_diameter // 2,
                                            c.y + self.checker_diameter // 2 + self.checker_diameter * checkers,
                                            self.checker_diameter // 2, color=color, batch=batch,
                                            group=group)
            else:
                raise ValueError("player has to be BLACK/WHITE BUT ITS {}".format(player))

        assert 0 <= pos < NUM_POINTS, "positon should be onboard got position={}".format(pos)
        c = self.points_coord[pos]
        (checkers, player_color_at_pos) = board[pos]
        # print(checkers,"=,checkers for pos =",pos)
        checkers = checkers + (0 if is_target else -1)
        checkers = clamp(x=checkers, l=0, u=4)
        # assert player_color_at_pos == player,""
        x = c.x
        y = c.y
        x += self.checker_diameter // 2
        if pos >= 12:
            y -= (self.checker_diameter // 2)
            y -= self.checker_diameter * checkers
        else:
            y += (self.checker_diameter // 2)
            y += self.checker_diameter * checkers
        return pyglet.shapes.Circle(x, y, self.checker_diameter // 2,
                                    color=color, batch=batch, group=group)

    def render(self, board, bar, off, state_w, state_h, agent, return_rgb_array=True, action=None):
        # assert agent is not None
        gl.glClearColor(1, 1, 1, 1)
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.clear()
        batch = pyglet.graphics.Batch()
        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)
        sprites = []

        for point, (checkers, player) in enumerate(board):

            if player is not None:
                assert player in [WHITE, BLACK], print("Should be WHITE (0) or BLACK (1), not {}".format(player))
                assert point in self.points_coord, print("Should be 0 <= point < 24, not {}".format(point))
                assert 0 <= checkers <= 15, print("Should be 0 <= checkers <= 15, not {}".format(checkers))

                c = self.points_coord[point]
                img = self.checkers[player][checkers]
                checkers = 5 if checkers > 5 else checkers
                img.width = self.checker_diameter
                img.height = self.checker_diameter * checkers

                if point < 12:
                    s = pyglet.sprite.Sprite(img=img, x=c.x, y=c.y, batch=batch, group=background)
                else:
                    s = pyglet.sprite.Sprite(img=img, x=c.x + self.checker_diameter,
                                             y=c.y - (checkers * self.checker_diameter) + img.height, batch=batch,
                                             group=background)
                    s.rotation = 180

                sprites.append(s)

        for player in [WHITE, BLACK]:
            # BAR
            checkers = bar[player]
            if checkers > 0:
                c = self.points_coord['bar_{}'.format(player)]

                img = self.checkers[player][checkers]
                checkers = 5 if checkers > 5 else checkers
                img.width = self.checker_diameter
                img.height = self.checker_diameter * checkers

                if player == BLACK:
                    s = sprites.append(pyglet.sprite.Sprite(img=img, x=c.x, y=c.y, batch=batch, group=background))
                else:
                    s = pyglet.sprite.Sprite(img=img, x=c.x + self.checker_diameter,
                                             y=c.y - (checkers * self.checker_diameter) + img.height, batch=batch,
                                             group=background)
                    s.rotation = 180
                sprites.append(s)

            # OFF
            checkers = off[player]
            if checkers > 0:

                c = self.points_coord['off_{}'.format(player)]
                img = self.checkers[player][checkers]
                checkers = 5 if checkers > 5 else checkers
                img.width = self.checker_diameter
                img.height = self.checker_diameter * checkers

                if player == WHITE:
                    s = sprites.append(pyglet.sprite.Sprite(img=img, x=c.x, y=c.y, batch=batch, group=background))
                else:
                    s = pyglet.sprite.Sprite(img=img, x=c.x + self.checker_diameter,
                                             y=c.y - (checkers * self.checker_diameter) + img.height, batch=batch,
                                             group=background)
                    s.rotation = 180

                sprites.append(s)

        gl.glViewport(0, 0, state_w, state_h)
        shapes = []
        if action is not None:
            action_from = action[0]
            # self, pos, board, bar, off, player, batch, group, color, is_target
            shape = self.render_action(pos=action_from, board=board, bar=bar, off=off, player=agent, batch=batch,
                                       group=foreground, color=(0, 0, 255), is_target=False)
            shapes.append(shape)
            action_to = action[1]
            shape = self.render_action(pos=action_to, board=board, bar=bar, off=off, player=agent,batch=batch,
                                       group=foreground, color=(0, 0, 255), is_target=True)
            shape.opacity=90

            shapes.append(shape)

            # c = self.points_coord['bar_{}'.format(BLACK)]
            # c2=self.points_coord[action_to]
            # print("black bar cordinates=", c)
            #
            # c2 = self.points_coord['bar_{}'.format(WHITE)]
            # # c2=self.points_coord[action_to]
            # print("white bar cordinates=", c2)
            #
            # c3 = self.points_coord['off_{}'.format(BLACK)]
            # print("black off cordinates=", c3)
            # c4 = self.points_coord['off_{}'.format(WHITE)]
            # print("white off cordinates=", c4)
            # for checkers in range(5):
            #     for p, _ in enumerate(board):
            #         c = self.points_coord[p]
            #         x = c.x
            #         y = c.y
            #         x += self.checker_diameter // 2
            #         if p >= 12:
            #             y -= (self.checker_diameter // 2)
            #             y -= self.checker_diameter * checkers
            #         else:
            #             y += (self.checker_diameter // 2)
            #             y += self.checker_diameter * checkers
            #         cur_shape = pyglet.shapes.Circle(x, y, self.checker_diameter // 2,
            #                                          color=(0, 0, 255), batch=batch, group=foreground)
            #         cur_shape.opacity = 128
            #         shapes.append(cur_shape)
            #     c = self.points_coord['bar_{}'.format(BLACK)]
            #     # c2=self.points_coord[action_to]
            #
            #     c2 = self.points_coord['bar_{}'.format(WHITE)]
            #     # c2=self.points_coord[action_to]
            #
            #     c3 = self.points_coord['off_{}'.format(BLACK)]
            #     c4 = self.points_coord['off_{}'.format(WHITE)]
            #
            #     shape = pyglet.shapes.Circle(c.x + self.checker_diameter // 2,
            #                                  c.y + self.checker_diameter // 2 + self.checker_diameter * checkers,
            #                                  self.checker_diameter // 2, color=(255, 0, 0), batch=batch,
            #                                  group=foreground)
            #     shapes.append(shape)
            #     shape2 = pyglet.shapes.Circle(c2.x + self.checker_diameter // 2,
            #                                   c2.y - self.checker_diameter // 2 - self.checker_diameter * checkers,
            #                                   self.checker_diameter // 2, color=(0, 255, 255), batch=batch,
            #                                   group=foreground)
            #
            #     shape3 = pyglet.shapes.Circle(c3.x + self.checker_diameter // 2,
            #                                   c3.y - self.checker_diameter // 2 - self.checker_diameter * checkers,
            #                                   self.checker_diameter // 2, color=(255, 0, 255), batch=batch,
            #                                   group=foreground)
            #
            #     shape4 = pyglet.shapes.Circle(c4.x + self.checker_diameter // 2,
            #                                   c4.y + self.checker_diameter // 2 + self.checker_diameter * checkers,
            #                                   self.checker_diameter // 2, color=(255, 255, 0), batch=batch,
            #                                   group=foreground)
            #     shapes.append(shape2)
            #     shapes.append(shape3)
            #     shapes.append(shape4)

        pyglet.sprite.Sprite(img=self.empty_board_image, batch=None).draw()
        batch.draw()

        arr = None

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
            arr = arr.reshape((state_h, state_w, 4))
            arr = arr[::-1, :, 0:3]

        gl.glViewport(0, 0, SCALING * self.window.width, SCALING * self.window.height)
        pyglet.sprite.Sprite(img=self.empty_board_image, batch=None).draw()
        batch.draw()

        self.window.flip()

        return arr
