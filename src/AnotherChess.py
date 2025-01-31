from math import trunc
from sqlite3 import IntegrityError
from Board import *
from Graphics import *
from Sound import *
from Client import *
from enum import Enum
import pygame, sys
import pygame_gui
import threading

pygame.init()
size = WIDTH, HEIGHT = 700, 625
SQUARE_SIZE = 60
black = 0, 0, 0

piece = 0
assets = loadAssets(SQUARE_SIZE)
gameState = Enum('gameState', ['REFRESH', 'STANDBY', 'PICKUP', 'HOLDPIECE', 'PUTDOWN', 'PROMOTE'])

currentState = gameState.REFRESH

screen = pygame.display.set_mode(size)
pygame.display.set_caption("AnotherChess Client")
screen.fill((255, 255, 255))
manager = pygame_gui.UIManager((WIDTH, HEIGHT), 'src/theme.json')
clock = pygame.time.Clock()
board = Board()
mixer = SoundMixer()
network = Client()
board.client = network
guiButtons = loadGuiButtons(manager)

threadQueue = []        # mutable data type for keeping track of data between threads in different files

promoteButtons = 0
promotePiece = 0

menuTable = createMenu(board, guiButtons)

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT: sys.exit()
        if currentState == gameState.PROMOTE:
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                for i, button in enumerate(promoteButtons):
                    if event.ui_element == button:
                        m = board.tryMove(board.heldPiece, promotePiece, i)
                        if m: 
                            updateTable(menuTable)
                            mixer.playMove(m[0], m[1])
                            if m[2]: guiButtons[1].enable()
                        piece = 0
                        for p in promoteButtons: p.kill()
                        promoteButtons = 0
                        board.placePiece()
                        currentState = gameState.REFRESH
                        break
            manager.process_events(event)
            continue
        if event.type == pygame.MOUSEBUTTONDOWN: 
            pos = pygame.mouse.get_pos()
            coords = calculateSquare(pos, board, SQUARE_SIZE)
            if (coords[0] < 0 or coords[0] > 7 or coords[1] < 0 or coords[1] > 7 or not board.isCurrentMove or (board.isOnline and not board.onlineTurn)):
                currentState = gameState.STANDBY
            else: currentState = gameState.PICKUP
        if event.type == pygame.MOUSEBUTTONUP: currentState = gameState.PUTDOWN
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == guiButtons[0]:       # flip
                board.flip()
            elif event.ui_element == guiButtons[1]:     # new game
                for i in range(1, 4):
                    guiButtons[i].disable()
                menuTable.clear()
                menuTable.force_surface_update()
                board.reset()
            elif event.ui_element == guiButtons[2]:     # forward
                if board.browseForward(): guiButtons[2].disable()   # think of more elegant solution
                guiButtons[3].enable()
                menuTable.select_widget(None)
                menuTable.select_widget(str(board.ply))
            elif event.ui_element == guiButtons[3]:     # back
                if board.browseBack(): guiButtons[3].disable()
                guiButtons[2].enable()
                if (board.ply):
                    menuTable.select_widget(None) 
                    menuTable.select_widget(str(board.ply))
            elif event.ui_element == guiButtons[4]:
                guiButtons[4].disable()
                searchGameThread = threading.Thread(target = network.searchGame, args=(10, 0), daemon=True)
                searchGameThread.start()
                board.isOnline = True
                eventStreamThread = threading.Thread(target = network.eventStream, args=(board, menuTable, mixer, threadQueue), daemon=True)
                eventStreamThread.start()
            elif event.ui_element == guiButtons[5]:
                if board.isOnline: 
                    network.clientResign()
                    guiButtons[1].enable()
            currentState == gameState.REFRESH
        manager.process_events(event)

    if len(threadQueue):            # maybe support multiple items in queue in future, as needed
        if threadQueue[0] == 1:
            currentState = gameState.REFRESH
            threadQueue.clear()


    if (currentState == gameState.PICKUP):
        if (board.board[coords[0]][coords[1]]): 
            piece = board.pickupPiece(coords)
        currentState = gameState.HOLDPIECE

    elif (currentState == gameState.HOLDPIECE):
        if (piece):
            pos = pygame.mouse.get_pos()
            printBoard(screen, assets, board, SQUARE_SIZE)
            adjustedPos = list(pos)
            inBounds(adjustedPos, SQUARE_SIZE)
            screen.blit(assets[piece], (adjustedPos[0] - SQUARE_SIZE/2, adjustedPos[1] - SQUARE_SIZE/2))

    elif (currentState == gameState.PUTDOWN):
        if (piece):
            pos = pygame.mouse.get_pos()
            coords = calculateSquare(pos, board, SQUARE_SIZE)
            if (coords[0] < 0 or coords[0] > 7 or coords[1] < 0 or coords[1] > 7):
                pass
            elif (piece % 10 == 1 and (coords[1] == 0 or coords[1] == 7)):
                if board.availableMoves(board.heldPiece[0], board.heldPiece[1])[coords[0]][coords[1]] and not board.moveInCheck(board.heldPiece, coords):
                    printBoard(screen, assets, board, SQUARE_SIZE)
                    currentState = gameState.PROMOTE
                    promoteButtons = generateButtons(manager, board, coords, SQUARE_SIZE)
                    promotePiece = coords
            else:
                m = board.tryMove(board.heldPiece, coords, None)
                if m:
                    updateTable(menuTable)
                    mixer.playMove(m[0], m[1])
                    if m[2]: guiButtons[1].enable()
                    guiButtons[3].enable()
        if not promoteButtons:
            piece = 0
            board.placePiece()
            currentState = gameState.REFRESH

    elif (currentState == gameState.PROMOTE):
        time_delta = clock.tick(60)/1000.0
        manager.update(time_delta)
        manager.draw_ui(screen)

    elif (currentState == gameState.REFRESH):
        printBoard(screen, assets, board, SQUARE_SIZE)
        currentState = gameState.STANDBY

    elif (currentState == gameState.STANDBY):
        time_delta = clock.tick(60)/1000.0
        manager.update(time_delta)
        manager.draw_ui(screen)

    menuTable.update(events)
    menuTable.draw(screen)
    pygame.display.update()