import camera_control as cc
import bg_board_cv as bgcv
import cv2
import time

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def test_bgcv_end_to_end():

    cv_obj = bgcv.BackgammonCV()

    status = cv_obj.board_init(output_img_with_containers=True)

    if (status.return_value != True):

        print(status.error_message)
        while (True):
            cv2.imshow("Error image", status.output_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    x = input("Board initialization completed successfully. Please place pieces and press 's' for capturing board status.")
    if (x == 's'):

        status, board, bar = cv_obj.get_current_board_status()
        if (status.return_value == False):
            print(status.error_message)
            while (True):
                cv2.imshow("Error image", status.output_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

        print("Here are the pieces detected on the board: {0}\nbar: {1}".format(board, bar))


def test_checkers_detection():

    camera = cc.init_camera()


    bg_obj = bgcv.BackgammonCV()
    while (True):

        total_attempts = 0
        cropped_img_list = []
        for i in range(bgcv.NUM_IMAGES_FOR_CHECKERS_DETECTION):


            success = False
            while (success != True):

                img = cc.take_image(camera)
                total_attempts += 1
                cv2.imshow('Original image', img)

                retval = bgcv.board_detection_green_dots(img)
                if (retval.return_value == True):
                    cropped_img_list.append(retval.output_image)
                    success = True
                #else:
                    #print(retval.error_message)

            #time.sleep(bgcv.TIME_DELAY_BETWEEN_ATTEMPTS)

        print ("Captured {0} images in {1} attempts".format(len(cropped_img_list), total_attempts))

        white, black, retval = bg_obj.checkers_detection(cropped_img_list, debug=True)
        if (white is not None):
            retval.output_image = bgcv.print_circles_on_image(retval.output_image, white)
            print ("white = {0}".format(white.shape[1]))
        if (black is not None):
            retval.output_image = bgcv.print_circles_on_image(retval.output_image, black, (0, 255, 0))
            print("black = {0}".format(black.shape[1]))

        cv2.imshow('cropped board', retval.output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cc.release_camera(camera)



def test_class():

    cv_obj = bgcv.BackgammonCV()

    while (True):

        status = cv_obj.board_init(output_img_with_containers=True)
        if (status.return_value == True):
            print("Success!")
            cv2.imshow("Cropped board with containers", status.output_image)
        else:
            print(status.error_message)
            cv2.imshow("Error image", status.output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def test_board_detection():

    camera = cc.init_camera()

    while (True):

        img = cc.take_image(camera)
        cv2.imshow('Original image', img)

        board = bgcv.board_detection_green_dots(img, debug=True).output_image
        if (board.shape[0] > 0):
            cv2.imshow('Detected Board', board)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    cc.release_camera(camera)

def test_board_init():

    camera = cc.init_camera()

    while (True):

        img = cc.take_image(camera)
        cv2.imshow('Original image', img)

        board = bgcv.board_detection_green_dots(img).output_image

        if (board.shape[0] > 0):
            cv2.imshow('Detected Board', board)
            ret_tri = bgcv.get_red_triangles_contours(board).output_list

            if (len(ret_tri) == 12):
                #cv2.imshow('Red triangles', cv2.drawContours(board, ret_tri, -1, (0, 255, 0), 2))
                ret_rect = bgcv.get_checkers_containers(ret_tri)

                for i in range(len(ret_rect)):
                    if (ret_rect[i] != None):
                        cv2.rectangle(board, ret_rect[i][0:2],
                                      (ret_rect[i][0] + ret_rect[i][2], ret_rect[i][1] + ret_rect[i][3]), (0, 255, 0), 2)

                        # Put cell name inside the rectangle center
                        #M = cv2.moments(ret_rect[i])
                        #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        center = (ret_rect[i][0] + ret_rect[i][2] // 2, ret_rect[i][1] + ret_rect[i][3] // 2)
                        cv2.putText(board, "{0}".format(i), center, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.3, color=(0, 255, 0))

                cv2.imshow('Containers', board)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break






# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #test_bgcv_end_to_end()

    #test_checkers_detection()

    #test_class()

    #test_board_detection()

    test_board_init()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
