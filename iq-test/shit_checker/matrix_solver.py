import cv2
import numbpy as np


def check_matrix(img_list, choices):
  pass

def find_shapes_in_image(img, debug=False):
  # find contours
  contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # draw contours
  img_contours = cv2.drawContours(np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8), contours, -1, (255, 255, 255), 2)

  if debug:
    cv2.imshow('contours', img_contours)

  # create list of contours
  contours_list = []
  for contour in contours:
    contours_list.append(contour)
  # sort contours
  contours_list = sorted(contours_list, key=lambda x: len(x), reverse=True)

  # find squares
  squares = []
  for contour in contours_list:
    # find bounding box
    x, y, w, h = cv2.boundingRect(contour)
    # check if square
    if (w > h and w < img.shape[1]/2):
      # check if it is not a small square
      if (w > img.shape[0]/4):
        # check if it is not a small square
        if (w > img.shape[1]/4):
          # draw square
          img_square = cv2.rectangle(np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8), (x, y), (x+w, y+h), (255, 255, 255), 2)
          # check if it is not a small square
          if (w > img.shape[1]/8):
            # add square to squares list
            squares.append((x, y, w, h))
            # draw square