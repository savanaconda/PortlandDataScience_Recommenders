# PortlandDataScience_Recommenders
work on recommendation systems in the Portland Data Science Meetup series

System Requirements: Python 3.6.2, Pandas 0.20.3, Matplotlib 2.0.2, Numpy 1.13.3

recommenders.py is the file with the main code which I wrote
recommendation_test.py is not written by savanaconda, it is a collaboration with a fellow data science meetup attendee

The goal of this was to use ratings from boardgamegeeks.com to recommend and recommends board games to
users based on their preferences. Implements user-based collaborative filtering to group similar users
based on their ratings and swap recommendations for those users.

Limitations to this algorithm are that it does not handle users which have very few ratings or do not group
well with other users, and it also can only recommend certain games to users rather than giving a probability that
a user will like a given game
