import cv2
import camera_control as cc
import numpy as np
import scipy.spatial as spt
import time
import gym_backgammon.envs.backgammon as bg_game

TRIANGLE_RATIO = 4.5
TRIANGLE_RATIO_DELTA = 2
LARGE_EDGES_RATIO_MIN = 0.9
LARGE_EDGES_RATIO_MAX = 1.1
RED_TRIANGLES_EXPECTED = 12
SINGLE_COLOR_TRIANGLES_PER_QUARTER = 3
SQUARE_EDGES_RATIO_DELTA = 0.1

NUM_ATTEMPTS = 10
TIME_DELAY_BETWEEN_ATTEMPTS = 0.0001
NUM_IMAGES_FOR_CHECKERS_DETECTION = 11

# Constants for detecting the green board markers
MIN_EDGE_MARKER_DISTANCE = 100
BOARD_EDGE_RATIO = 1
BOARD_EDGE_DIAGONAL_RATIO = np.sqrt(2)
DELTA_RATIOS = 0.04


class Status:

    def __init__(self, return_value=True, error_message="", output_image=np.ndarray([]), output_list=[]):
        self.return_value = return_value
        self.error_message = error_message
        self.output_image = output_image
        self.output_list = output_list


class BackgammonCV:

    def __init__(self):

        self.camera = cc.init_camera()
        self.checkers_containers = None

        # Checkers HSV color ranges (lower and upper range per checker type)
        # self.white_checkers_lower_HSV = np.array([0, 30, 120])
        # self.white_checkers_upper_HSV = np.array([65, 210, 255])
        #self.black_checkers_lower_HSV = np.array([95, 67, 108])
        #self.black_checkers_upper_HSV = np.array([108, 123, 220])
        self.white_checkers_lower_HSV = np.array([0, 25, 100])
        self.white_checkers_upper_HSV = np.array([20, 150, 180])
        #self.black_checkers_lower_HSV = np.array([105, 42, 77])
        #self.black_checkers_upper_HSV = np.array([120, 100, 255])
        # daylight
        self.black_checkers_lower_HSV = np.array([107, 46, 91])
        self.black_checkers_upper_HSV = np.array([112, 255, 255])


    def board_init(self, output_img_with_containers=False):
        '''
        This function performs the initial board calibration, including getting all the checkers containers and storing
        them in a class variable
        :param output_img_with_containers: for debug, print the output image with the checkers containers on top of it
        :return: Status object
        '''

        # Apply board detection
        for i in range(NUM_ATTEMPTS):

            # Take picture from the camera
            curr_image = cc.take_image(self.camera)
            cropped_board_image = None

            ret_status = board_detection_green_dots(curr_image)
            if (ret_status.return_value == True):
                cropped_board_image = ret_status.output_image

                ret_status = get_red_triangles_contours(cropped_board_image)
                if (ret_status.return_value == True):
                    self.checkers_containers = get_checkers_containers(ret_status.output_list)
                else:
                    time.sleep(TIME_DELAY_BETWEEN_ATTEMPTS)
            else:
                time.sleep(TIME_DELAY_BETWEEN_ATTEMPTS)
                continue

        # If board cropping failed after few attempts and delays, return error
        if (ret_status.return_value != True):
            return ret_status

        # Create an image with the visible checkers containers if needed (mostly for debug mode)
        output_image = np.ndarray([])
        if (output_img_with_containers == True):
            num_containers = len(self.checkers_containers)
            for i in range(num_containers):
                cv2.rectangle(cropped_board_image, self.checkers_containers[i][0:2],
                              (self.checkers_containers[i][0] + self.checkers_containers[i][2],
                               self.checkers_containers[i][1] + self.checkers_containers[i][3]),
                              (0, 255, 0), 2)
            output_image = cropped_board_image

        return Status(return_value=True, output_image=output_image)

    def checkers_detection(self, cropped_board_img_list, debug=False):
        '''
        Detects all checkers on the Backgammon board.
        :param cropped_board_img_list: a list of images, all of them tightly cropped images of the board.
        :param debug: If true, show debug image.
        :return: Two lists of circles, representing the white and black checkers detected on the board.
        '''

        num_images = len(cropped_board_img_list)
        white_checkers_list = []
        black_checkers_list = []
        for i in range(num_images):

            cropped_board_img = cropped_board_img_list[i]

            # Turn image to HSV for color separation
            img_hsv = cv2.cvtColor(cropped_board_img, cv2.COLOR_BGR2HSV)

            # Blur for better edge detection
            #img_hsv = cv2.medianBlur(img_hsv, 9)
            #img_hsv = cv2.GaussianBlur(img_hsv, (3, 3), 0)

            # Create mask for each checker type
            white_mask = cv2.inRange(img_hsv, self.white_checkers_lower_HSV, self.white_checkers_upper_HSV)
            black_mask = cv2.inRange(img_hsv, self.black_checkers_lower_HSV, self.black_checkers_upper_HSV)

            white_mask = cv2.medianBlur(white_mask, 5)
            #black_mask = cv2.GaussianBlur(black_mask, (5,5), 0)
            black_mask = cv2.medianBlur(black_mask, 3)

            # Try erosion and dilation
            #kernel = np.ones((3, 3), np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            w_eroded = cv2.erode(white_mask, kernel, iterations=1)
            w_dilated = cv2.dilate(w_eroded, kernel, iterations=2)

            black_kernel = np.ones((3, 3), np.uint8)
            b_eroded = cv2.erode(black_mask, kernel, iterations=1)
            b_dilated = cv2.dilate(b_eroded, kernel, iterations=1)

            if (debug == True):
                cv2.imshow('white checkers', w_dilated)
                cv2.imshow('black checkers', b_dilated)

            # Find the checkers' circles using Hough transform, on the corresponding mask for each color
            white_checkers_list.append(cv2.HoughCircles(w_dilated, cv2.HOUGH_GRADIENT, 1, 15,
                                              param1=250, param2=7, minRadius=7, maxRadius=15))
            black_checkers_list.append(cv2.HoughCircles(b_dilated, cv2.HOUGH_GRADIENT, 1, 15,
                                              param1=250, param2=7, minRadius=7, maxRadius=15))

        # Now we compare the number of circles we got in each image, and get votes for the number of circles in each
        # image
        checkers_count = np.zeros((num_images, 2), dtype=int)
        for i in range(num_images):

            if (white_checkers_list[i] is not None):
                checkers_count[i][bg_game.WHITE] = white_checkers_list[i].shape[1]
            else:
                checkers_count[i][bg_game.WHITE] = 0

            if (black_checkers_list[i] is not None):
                checkers_count[i][bg_game.BLACK] = black_checkers_list[i].shape[1]
            else:
                checkers_count[i][bg_game.BLACK] = 0

        max_vote_white = np.argmax(np.bincount(checkers_count[:, bg_game.WHITE]))
        max_vote_black = np.argmax(np.bincount(checkers_count[:, bg_game.BLACK]))

        out_ind_white = np.where(checkers_count[:, bg_game.WHITE] == max_vote_white)[0]
        out_ind_black = np.where(checkers_count[:, bg_game.BLACK] == max_vote_black)[0]

        return white_checkers_list[out_ind_white[0]], black_checkers_list[out_ind_black[0]], Status(True, output_image=cropped_board_img_list[0])

    def find_container_for_checker(self, checker_circle):
        '''
        Finds the checker container where this checker circle is currently appearing.
        :param checker_circle: The circle of this checker as detected by the camera.
        :return: The index of the relevant container, or -1 if no container was found.
        '''

        len_containers = len(self.checkers_containers)
        for i in range(len_containers):

            cont_x = self.checkers_containers[i][0]
            cont_y = self.checkers_containers[i][1]
            cont_w = self.checkers_containers[i][2]
            cont_h = self.checkers_containers[i][3]
            if (point_in_rectangle((cont_x, cont_y), cont_w, cont_h, checker_circle[0:2]) == True):
                return i

        return -1

    def get_current_board_status(self):
        '''

        :return: Returns an object representing the current status of the Backgammon board
        '''

        # Apply board detection, create several images of cropped board for getting accurate results on checkers
        # detection
        cropped_board_images = []
        for i in range (NUM_IMAGES_FOR_CHECKERS_DETECTION):

            success = False
            attempts = 0
            print('i = {0}, attempts = {1}'.format(i, attempts))

            while (success != True):

                # Take picture from the camera
                curr_image = cc.take_image(self.camera)
                attempts += 1

                ret_status = board_detection_green_dots(curr_image)
                if (ret_status.return_value == True):
                    cropped_board_images.append(ret_status.output_image)
                    success = True
                    #time.sleep(TIME_DELAY_BETWEEN_ATTEMPTS)
                    #time.sleep(TIME_DELAY_BETWEEN_ATTEMPTS)

        if (len(cropped_board_images) < NUM_IMAGES_FOR_CHECKERS_DETECTION):
            return Status(False,
                          error_message="Could not crop enough board images for checkers detection, got {0} images instead of {1}".format(len(cropped_board_images), NUM_IMAGES_FOR_CHECKERS_DETECTION)), None, None, None


        # Detect the checkers on the cropped board
        white_checkers, black_checkers, retval = self.checkers_detection(cropped_board_images)

        # Initialize the board and bar objects to be returned by the function
        board = [(0, None)] * bg_game.NUM_POINTS
        bar = [0, 0]

        if (white_checkers is None):
            num_white = 0
        else:
            num_white = white_checkers.shape[1]

        if (black_checkers is None):
            num_black = 0
        else:
            num_black = black_checkers.shape[1]

        print('Found {0} white and {1} black'.format(num_white, num_black))

        # Populate the board object with all white checkers
        for i in range(num_white):

            # Find the container for this checker
            curr_cont = self.find_container_for_checker(white_checkers[0][i])
            if (curr_cont != -1):

                if (curr_cont > 0):

                    # We match the indexes as container numbering goes 1-24 in the bgcv list
                    board[curr_cont - 1] = (board[curr_cont - 1][0] + 1, bg_game.WHITE)
                else:

                    # Container 0 is the bar
                    bar[bg_game.WHITE] += 1
            else:
                error_img = cv2.circle(cropped_board_images[0], (white_checkers[0][i][0], white_checkers[0][i][1]),
                                       int(white_checkers[0][i][2]), (0, 255, 0), 2)
                cv2.imshow("error", error_img)
                return Status(False,
                              error_message="Found a white checker that doesn't belong in any container, index = {0}.".format(
                                  i),
                              output_image=cropped_board_images[0]), None, None, None

        # Populate the board object with all black checkers
        for i in range(num_black):

            # Find the container for this checker
            curr_cont = self.find_container_for_checker(black_checkers[0][i])

            if (curr_cont != -1):

                if (curr_cont != 0):

                    # First we make sure there are no white checkers populated in this container
                    if (board[curr_cont - 1][0] == 0):

                        # We match the indices as container numbering goes 1-24 in the bgcv list
                        board[curr_cont - 1] = (board[curr_cont - 1][0] + 1, bg_game.BLACK)
                    elif (board[curr_cont - 1][1] == bg_game.BLACK):
                        board[curr_cont - 1] = (board[curr_cont - 1][0] + 1, bg_game.BLACK)
                    else:
                        return Status(False,
                                      error_message="Found black and white checker together in the same container, container is {0}".format(
                                          curr_cont),
                                      output_image=cropped_board_images[0]), None, None, None

                else:
                    # Container 0 is the bar
                    bar[bg_game.BLACK] += 1
            else:
                return Status(False,
                              error_message="Found a black checker that doesn't belong in any container, index = {0}.".format(
                                  i),
                              output_image=cropped_board_images[0]), None, None, None


        sum_white = bar[bg_game.WHITE]
        sum_black = bar[bg_game.BLACK]
        for (checkers, player) in board:
            if player == bg_game.BLACK:
                sum_black += checkers
            if player == bg_game.WHITE:
                sum_white += checkers
        off = [15, 15]
        off[bg_game.BLACK] -= sum_black
        off[bg_game.WHITE] -= sum_white
        return Status(True, output_image=cropped_board_images[0]), board, bar, off


def point_in_rectangle(top_left_rect_point, rect_width, rect_height, test_point):

    x = test_point[0]
    y = test_point[1]

    # Check x axis
    if (x > top_left_rect_point[0] and x < top_left_rect_point[0] + rect_width):

        # Check y axis
        if (y > top_left_rect_point[1] and y < top_left_rect_point[1] + rect_height):
            return True

    return False


def board_detection_green_dots(input_img, debug=False):
    '''

    :param input_img: Input image as captured by the camera. The image must include the Backgommon board, with four
    green stickers on its vertices
    :param debug: If true, show debug image output
    :return: a 400x400 aligned, tightly cropped board
    '''

    # Turn image to HSV for green color separation
    img_hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

    # img_hsv = cv2.medianBlur(img_hsv, 13)
    # img_hsv = cv2.GaussianBlur(img_hsv, (21, 21), 0)

    # Define HSV masks for green color
    #lower_green = np.array([40, 40, 40])
    #upper_green = np.array([70, 255, 255])
    lower_green = np.array([34, 49, 40])
    upper_green = np.array([83, 255, 255])

    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ero_dil = cv2.erode(green_mask, kernel, iterations=1)
    ero_dil = cv2.dilate(ero_dil, kernel, iterations=2)

    # Debug print
    if (debug == True):
        cv2.imshow('Green HSV mask & dilation', ero_dil)

    # Find the green circle markers representing the board vertices using Hough transform
    circles = cv2.HoughCircles(ero_dil, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=70, param2=10, minRadius=2, maxRadius=15)

    # We need to find exactly 4 green circles, if not -- the function fails and returns an empty array, the calling
    # function should reattempt
    if circles is None:
        return Status(False, "No circles found in the image", green_mask)
    elif (circles.shape[1] < 4):
        return Status(False,
                      "Too small number of circles in the image, found {0}, at least 4 expected.".format(
                          circles.shape[1]),
                      print_circles_on_image(input_img, circles))
    else:
        # Find the right 4 circles at the vertices of the board: their centers are quite far away from each other, and
        # the distances to two of them are equal (these are the square edges), and the distance to the third is
        # sqrt(2) times the edge length

        # First, calculate distance matrix of all circle centers
        num_circles_found = circles.shape[1]
        distance_matrix = np.zeros((num_circles_found, num_circles_found))
        filtered_circles = np.zeros((1, 4, 3))
        found_match = False
        for i in range(num_circles_found):

            for j in range(num_circles_found):

                if ((i != j) and (distance_matrix[i, j] == 0)):
                    distance_matrix[i, j] = np.linalg.norm(circles[0][i][0:2] - circles[0][j][0:2])
                else:
                    # Continue as we already computed this distance
                    continue

        # Filter the circles according to the distances we expect to see between them
        for i in range(num_circles_found):

            sorted_distances = np.sort(distance_matrix[i])
            for j in range(num_circles_found - 2):

                e1 = sorted_distances[j]
                e2 = sorted_distances[j + 1]

                if (e1 < MIN_EDGE_MARKER_DISTANCE):
                    continue
                if (e2 / e1 < BOARD_EDGE_RATIO + DELTA_RATIOS):
                    # We probably found the two points on the edges, now need to find the points on the diagonal
                    for k in range(j + 2, num_circles_found):
                        diag_normalized = sorted_distances[k] / BOARD_EDGE_DIAGONAL_RATIO
                        avg_edge = (e1 + e2) / 2
                        ratio = np.abs(1 - diag_normalized / avg_edge)
                        if (ratio < DELTA_RATIOS):
                            # We found the third point on the diagonal!
                            found_match = True
                            filtered_circles[0][0] = circles[0][i]
                            # print(np.where(distance_matrix[i] == e1)[0])
                            # print(circles[0][np.where(distance_matrix[i] == e1)[0]])
                            # print(e1, e2)
                            filtered_circles[0][1] = circles[0][np.where(distance_matrix[i] == e1)[0][0]]
                            filtered_circles[0][2] = circles[0][np.where(distance_matrix[i] == e2)[0][0]]
                            filtered_circles[0][3] = circles[0][
                                np.where(distance_matrix[i] == sorted_distances[k])[0][0]]
                            break

        if (found_match != True):
            return Status(False,
                          "{0} circles were found in the image, but no 4 of them has the board square pattern expected.".format(
                              num_circles_found),
                          print_circles_on_image(input_img, circles))

    # Get the pixels at the corners of the image
    top_left_corner = [0, 0]
    top_right_corner = [input_img.shape[1], 0]
    bottom_right_corner = [input_img.shape[1], input_img.shape[0]]
    bottom_left_corner = [0, input_img.shape[0]]

    # Init array for KD tree
    points_arr = np.zeros((4 + filtered_circles.shape[1], 2))
    points_arr[0] = top_left_corner
    points_arr[1] = top_right_corner
    points_arr[2] = bottom_right_corner
    points_arr[3] = bottom_left_corner

    for i in range(filtered_circles.shape[1]):
        points_arr[4 + i] = filtered_circles[0][i][0:2]

    # Use KD tree for locating each green point corresponding to its nearest image corner
    tree = spt.KDTree(points_arr)
    x, nearest_points = tree.query(points_arr[0:4], 2)

    top_left_src = points_arr[nearest_points[0][1]]
    top_right_src = points_arr[nearest_points[1][1]]
    bottom_right_src = points_arr[nearest_points[2][1]]
    bottom_left_src = points_arr[nearest_points[3][1]]

    # Now, we have all points we need for performing homography
    src_pts = np.array([top_left_src,
                        bottom_left_src,
                        bottom_right_src,
                        top_right_src])

    # Use homography for aligning the board on the camera, with 400x400 image of the board
    dst_pts = np.array([[0, 0],
                        [0, 399],
                        [399, 399],
                        [399, 0]], dtype="float32")

    h, status = cv2.findHomography(src_pts, dst_pts)

    out = cv2.warpPerspective(input_img, h, (400, 400))
    return Status(True, output_image=out)


def print_circles_on_image(input_image, circles, color=(0, 0, 255)):
    '''

    :param input_image: Image to print circles on
    :param circles: a list of cv2 circless
    :param color: BGR color
    :return: the output image with circles drawn
    '''
    num_circles = circles.shape[1]
    circles = circles.astype(int)
    out_image = input_image.copy()
    for i in range(num_circles):
        out_image = cv2.circle(out_image, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], color, 2)

    return out_image


def get_red_triangles_contours(input_img):
    '''
    This function returns a list of the contours representing the red triangles in the backgammon board.
    :param input_img: The input image of the board after tight cropping and homography
    :return: If successful, returns a list with 12 triangle contours representing the red triangles in the backgammon
    board. Otherwise, return empty list
    '''

    # Perform smoothing for better edge detection
    input_img = cv2.GaussianBlur(input_img, (7, 7), 0)

    # Turn image to HSV for red color separation
    img_hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

    # Define HSV masks for lower and upper red
    lower_red_1 = np.array([0, 100, 20])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([160, 100, 20])
    upper_red_2 = np.array([179, 255, 255])
    lower_mask = cv2.inRange(img_hsv, lower_red_1, upper_red_1)
    upper_mask = cv2.inRange(img_hsv, lower_red_2, upper_red_2)
    full_mask = lower_mask + upper_mask

    # Find all contours surrounding red shapes
    red_contours, hierarchy = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter all contours in the shape of a triangle, we need 12
    red_cnt = len(red_contours)
    red_triangles = []
    for i in range(red_cnt):

        # Approximate a polygon from each contour
        curr_contour = red_contours[i]
        epsilon = 0.05 * cv2.arcLength(curr_contour, True)
        approx = cv2.approxPolyDP(curr_contour, epsilon, True)

        # Look only at triangles
        cand_tri_avg_base = 0
        curr_tri = np.zeros((3))
        if (len(approx) == 3):

            # Capture all edges of the triangle, mark the base and the other edges (base is shortest)
            curr_tri[0] = np.linalg.norm(approx[2] - approx[1])
            curr_tri[1] = np.linalg.norm(approx[1] - approx[0])
            curr_tri[2] = np.linalg.norm(approx[2] - approx[0])
            curr_tri = np.sort(curr_tri)
            e1_base = curr_tri[1] / curr_tri[0]
            e2_base = curr_tri[2] / curr_tri[0]
            e2_e1_ratio = curr_tri[2] / curr_tri[1]
            min_thresh = TRIANGLE_RATIO - TRIANGLE_RATIO_DELTA
            max_thresh = TRIANGLE_RATIO + TRIANGLE_RATIO_DELTA

            # Capture only triangles with the correct ratios between edges as expected in the backgammon board
            if (e1_base > min_thresh and e1_base < max_thresh):
                if (e2_base > min_thresh and e2_base < max_thresh):
                    if (e2_e1_ratio > LARGE_EDGES_RATIO_MIN and e2_e1_ratio < LARGE_EDGES_RATIO_MAX):
                        red_triangles.append(approx)
                        cand_tri_avg_base += curr_tri[0]

    found_triangle_count = len(red_triangles)
    if (found_triangle_count == RED_TRIANGLES_EXPECTED):
        return Status(return_value=True, output_list=red_triangles)
    else:
        return Status(return_value=False,
                      error_message="Found {0} red triangles instead of 12 expected.".format(found_triangle_count),
                      output_image=input_img)

    return out


def get_checkers_containers(red_triangles_contours):
    '''
    This function returns a list with all rectangles where checkers can be placed during the game.
    :param red_triangles_contours: a list of 12 triangles contours corresponding to the red triangles on the backgammon
    board.
    :return: A list with 25 rectangle contours that represents all legit places on the board where checkers can be
    placed. This accounts for 24 triangles (indices 1-24) and the mid-board bar (index 0).
    '''

    # We need to sort red triangle contours from left to right and top to bottom
    # construct the list of bounding boxes and sort them from top to bottom

    # First, sort them top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in red_triangles_contours]
    (red_triangles_contours, boundingBoxes) = zip(*sorted(zip(red_triangles_contours, boundingBoxes),
                                                          key=lambda b: b[1][1], reverse=False))

    # Split the list to 6 top triangles and 6 bottom triangles
    top_tri = red_triangles_contours[0:6]
    top_boxes = boundingBoxes[0:6]
    bottom_tri = red_triangles_contours[6:12]
    bottom_boxes = boundingBoxes[6:12]

    # Order the top and bottom lists from left to right, separately
    (top_tri, top_boxes) = zip(*sorted(zip(top_tri, top_boxes), key=lambda b: b[1][0], reverse=False))
    (bottom_tri, bottom_boxes) = zip(*sorted(zip(bottom_tri, bottom_boxes), key=lambda b: b[1][0], reverse=False))

    # Preparing the returned list with 25 indices, filling them one by one. NOTE: There is some code duplication here,
    # Please read the section divisions and comments carefully before changing!
    out_list = [None] * 25

    top_boxes = list(top_boxes)

    LENGTH_FACTOR = 20
    for i in range(len(top_boxes)):
        curr_tuple = list(top_boxes[i])
        curr_tuple[3] += LENGTH_FACTOR
        top_boxes[i] = tuple(curr_tuple)

    ### Top left quarter (indices are top-left from the camera POV, which is opposite to the human player!)
    out_list[1] = top_boxes[0]
    out_list[2] = (top_boxes[0][0] + top_boxes[0][2], top_boxes[0][1],
                    top_boxes[1][0] - (top_boxes[0][0] + top_boxes[0][2]), top_boxes[0][3])
    out_list[3] = top_boxes[1]
    out_list[4] = (top_boxes[1][0] + top_boxes[1][2], top_boxes[1][1],
                    top_boxes[2][0] - (top_boxes[1][0] + top_boxes[1][2]), top_boxes[1][3])
    out_list[5] = top_boxes[2]

    # The last black triangle in the quarter, here we take the width from the opposite (bottom) red triangle
    out_list[6] = (top_boxes[2][0] + top_boxes[2][2], top_boxes[2][1],
                    bottom_boxes[2][2], top_boxes[2][3])



    ### Top right quarter (indices are top-right from the camera POV, which is opposite to the human player!)
    out_list[7] = top_boxes[3]
    out_list[8] = (top_boxes[3][0] + top_boxes[3][2], top_boxes[3][1],
                    top_boxes[4][0] - (top_boxes[3][0] + top_boxes[3][2]), top_boxes[3][3])
    out_list[9] = top_boxes[4]
    out_list[10] = (top_boxes[4][0] + top_boxes[4][2], top_boxes[4][1],
                    top_boxes[5][0] - (top_boxes[4][0] + top_boxes[4][2]), top_boxes[4][3])
    out_list[11] = top_boxes[5]

    # The last black triangle in the quarter, here we take the width from the opposite (bottom) red triangle
    out_list[12] = (top_boxes[5][0] + top_boxes[5][2], top_boxes[5][1],
                    bottom_boxes[5][2], top_boxes[5][3])

    ### Bottom left quarter (indices are bottom-left from the camera POV, which is opposite to the human player!)
    # The first black triangle in the quarter, here we take the width from the opposite (top) red triangle
    out_list[24] = (top_boxes[0][0], bottom_boxes[0][1], bottom_boxes[0][2], top_boxes[0][3])

    out_list[23] = bottom_boxes[0]
    out_list[22] = (bottom_boxes[0][0] + bottom_boxes[0][2], bottom_boxes[0][1],
                    bottom_boxes[1][0] - (bottom_boxes[0][0] + bottom_boxes[0][2]), bottom_boxes[0][3])
    out_list[21] = bottom_boxes[1]
    out_list[20] = (bottom_boxes[1][0] + bottom_boxes[1][2], bottom_boxes[1][1],
                   bottom_boxes[2][0] - (bottom_boxes[1][0] + bottom_boxes[1][2]), bottom_boxes[1][3])
    out_list[19] = bottom_boxes[2]

    ### Bottom right quarter (indices are bottom-right from the camera POV, which is opposite to the human player!)
    # The first black triangle in the quarter, here we take the width from the opposite (top) red triangle
    out_list[18] = (top_boxes[3][0], bottom_boxes[3][1], bottom_boxes[3][2], top_boxes[3][3])

    out_list[17] = bottom_boxes[3]
    out_list[16] = (bottom_boxes[3][0] + bottom_boxes[3][2], bottom_boxes[3][1],
                   bottom_boxes[4][0] - (bottom_boxes[3][0] + bottom_boxes[3][2]), bottom_boxes[3][3])
    out_list[15] = bottom_boxes[4]
    out_list[14] = (bottom_boxes[4][0] + bottom_boxes[4][2], bottom_boxes[4][1],
                   bottom_boxes[5][0] - (bottom_boxes[4][0] + bottom_boxes[4][2]), bottom_boxes[4][3])
    out_list[13] = bottom_boxes[5]

    ### The bar
    out_list[0] = (out_list[6][0] + out_list[6][2], out_list[6][1],
                   bottom_boxes[2][2], bottom_boxes[2][1] + bottom_boxes[2][3])

    # Finished updating the return list
    return out_list

