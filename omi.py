############## Python-OpenCV Playing Card Detector ###############
#
# Author: Ravindu Fernando
# Description: Python script to detect and identify playing cards


# Import necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import Cards


### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX


train_ranks = Cards.load_ranks("suits_ranks")
train_suits = Cards.load_suits("suits_ranks")


# Grab the image
image = cv2.imread(r'images/8cards_top.jpg',cv2.IMREAD_COLOR)
img_copy_function = image.copy()


# Pre-process image (gray, blur, and threshold it)
pre_proc = Cards.preprocess_image(image)


# Find and sort the contours of all cards in the image (query cards)
cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)


# If there are no contours, do nothing
if len(cnts_sort) != 0:

    # Initialize a new "cards" list to assign the card objects.
    # k indexes the newly made array of cards.
    cards = []
    k = 0

    # For each contour detected:
    for i in range(len(cnts_sort)):
        if (cnt_is_card[i] == 1):

            # Create a card object from the contour and append it to the list of cards.
            # preprocess_card function takes the card contour and contour and
            # determines the cards properties (corner points, etc). It generates a
            # flattened 200x300 image of the card, and isolates the card's
            # suit and rank from the image.
            cards.append(Cards.preprocess_card(cnts_sort[i],image))

            # Find the best rank and suit match for the card.
            cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)

            # Draw center point and match result on the image.
            image = Cards.draw_results(image, cards[k])
   
            k = k + 1

        

	    
        # Draw card contours on image (have to do contours all at once or
        # they do not show up properly for some reason)
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image,temp_cnts, -1, (255,0,0), 15)
        
        

    imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.imshow(imgRGB)
    plt.show()
    
  

    
        



