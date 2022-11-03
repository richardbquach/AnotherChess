# Another Chess Project
The goal of this project is to train an AI using a camera to determine the position of a chessboard. With this function comes many different applications such as AI, connectivity with Lichess, move suggestions, etc.

## Key Functionality
The enclosed program will keep track of the board. The current state of the board as viewed by the AI will be displayed on the board. As a move is made, the screen will indicate the legality of the move. Different gamemodes can be selected via a UI.

## Planned Features
- Stronger AI
- Saving your game, so that you can pick it up another time
- Time control + other settings
- Connectivity to Lichess

## Installation
While in the root directory, install required libraries by running

`pip install -r requirements.txt`

Then, you can run the application by running

`python3 AnotherChess/AnotherChess.py`

and if things are successful, you should see the following window (no current piece movement).
![image](https://user-images.githubusercontent.com/62521534/199654781-26044fc6-b5ff-43d7-8a91-43f533ad57cc.png)
