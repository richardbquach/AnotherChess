from Storage import *
import threading

class Board:
    board = []
    storage = GameStack()
    isCurrentMove = True
    flags = []
    heldPiece = 0
    turn = True
    flipped = False
    lastMove = 0
    ply = 0
    isOnline = False
    onlineTurn = False
    client = None

    #notated columns, rows on the chessboard. (2, 3) is the third column, fourth row (from the bottom left)
    #1 - 6 are white pawns, rooks, knights, bishops, queens, and king, respectively. 11-16 are the same for black pieces.

    def __init__(self):
        self.board = [[0 for i in range(8)] for j in range(8)] # reset members
        self.storage = GameStack()
        self.isCurrentMove = True
        self.flags = [False for i in range(6)]
        self.heldPiece = 0
        self.turn = True
        self.flipped = False
        self.lastMove = 0
        self.ply = 0
        self.isOnline = False
        self.onlineTurn = False
            
        for i in range(8):
            self.board[i][1] = 1
            self.board[i][6] = 11
            
            if i < 5:
                self.board[i][0] = i + 2
                self.board[i][7] = i + 12
            else:
                self.board[i][0] = 9 - i
                self.board[i][7] = 19 - i

    def reset(self):
        self.__init__()

    def debugBoard(self, b = board):
        for i in range(7, -1, -1):
            for j in range(8):
                print(b[j][i], end = '')
                if b[j][i]/10 < 1:
                    print(" ", end = '')
                print(" ", end = '')
            print()
        print()

    def browseBack(self):
        isStart = self.storage.iterateBack()
        self.isCurrentMove = False
        self.retrieveStorage()
        self.ply -= 1                               # consider making a function which does multiple moves at a time. (same for browseForward, ofc)
        if isStart: return True
        else: return False

    def browseForward(self):
        if (self.storage.iterateForward()): self.isCurrentMove = True
        self.retrieveStorage()
        self.ply += 1
        if self.isCurrentMove: return True
        return False

    def retrieveStorage(self):
        self.board = self.storage.convertBoard(self.board)
                    
    def pickupPiece(self, pos):
        self.heldPiece = pos
        return self.board[pos[0]][pos[1]]

    def placePiece(self):
        self.heldPiece = 0
                    
    def inBounds(self, col, row):
        return col >= 0 and col < 8 and row >= 0 and row < 8

    def flip(self):
        self.flipped = not self.flipped

    def checkMove(self, start, end):   # delete
        if start == end:
            return False
        
        if self.turn:
            if self.board[start[0]][start[1]]/10 >= 1:
                return False
        else:
            if self.board[start[0]][start[1]]/10 < 1:
                return False

    def move(self, start, end):             # move piece without checks
        t = self.board[start[0]][start[1]]
        self.board[start[0]][start[1]] = 0
        self.board[end[0]][end[1]] = t

    def isOccupied(self, col, row):         # returns 1 if enemy piece, 2 if friendly
        if self.board[col][row] == 0:
            return 0
        elif self.turn and self.board[col][row]/10 >= 1:
            return 1
        elif not self.turn and self.board[col][row]/10 < 1:
            return 1
        else:
            return 2

    def promote(self, i, coords):
        add = 0
        p = 0
        if not self.turn:
            add = 10
        if i == 0:
            p = 5 + add
        elif i == 1:
            p = 2 + add
        elif i == 2:
            p = 3 + add
        else:
            p = 4 + add

        self.board[coords[0]][coords[1]] = p
        return
            
    def availableMoves(self, col, row, flip = False):   # display available moves for piece, not accounting for check or special moves
        out = [[0 for i in range(8)] for j in range(8)]
        p = self.board[col][row]
        if not flip:
            if self.turn and p > 10: return out
            if not self.turn and p < 10: return out
        else:
            self.turn = not self.turn
        piece = p % 10
        
        if piece == 1:        #PAWN
            direction = 1
            if not self.turn:
                direction = -1

            newRow = row + direction

            if self.inBounds(col, newRow) and not self.isOccupied(col, newRow) and not flip:
                out[col][row+direction] = 1

            for i in (-1, 1):
                if self.inBounds(col + i, newRow) and (self.isOccupied(col + i, newRow) == 1 or flip):
                    out[col+i][newRow] = 1

            if self.turn:
                if row == 1:
                    if not self.isOccupied(col, newRow + 1) and not flip:
                        out[col][newRow + 1] = 1
            else:
                if row == 6:
                    if not self.isOccupied(col, newRow - 1) and not flip:
                        out[col][newRow - 1] = 1
                            
        if piece == 2 or piece == 5:        #ROOK/QUEEN
            direction = [[col,row, i] for i in range(4)] #four directional checking
            while len(direction) > 0:
                for i in range(len(direction)):
                    d = direction[i]
                    if d[2] == 0:           #update checked square
                        d[1] += 1
                    elif d[2] == 1:
                        d[1] -= 1
                    elif d[2] == 2:
                        d[0] += 1
                    else:
                        d[0] -= 1

                    if not self.inBounds(d[0], d[1]) or self.isOccupied(d[0], d[1]) == 2:
                        direction.pop(i)
                        break
                    elif self.isOccupied(d[0], d[1]) == 1:
                        out[d[0]][d[1]] = 1
                        direction.pop(i)
                        break
                    else:
                        out[d[0]][d[1]] = 1

        if piece == 4 or piece == 5:        #BISHOP/QUEEN
            direction = [[col,row, i] for i in range(4)] #four directional checking
            while len(direction) > 0:
                for i in range(len(direction)):
                    d = direction[i]
                    if d[2] == 0:           #update checked square
                        d[0] += 1
                        d[1] += 1
                    elif d[2] == 1:
                        d[0] += 1
                        d[1] -= 1
                    elif d[2] == 2:
                        d[0] -= 1
                        d[1] += 1
                    else:
                        d[0] -= 1
                        d[1] -= 1

                    if not self.inBounds(d[0], d[1]) or self.isOccupied(d[0], d[1]) == 2:
                        direction.pop(i)
                        break
                    elif self.isOccupied(d[0], d[1]) == 1:
                        out[d[0]][d[1]] = 1
                        direction.pop(i)
                        break
                    else:
                        out[d[0]][d[1]] = 1

        if piece == 3:          #KNIFE
            o1 = [-1, 1]
            o2 = [-2, 2]

            for i in o1:
                for j in o2:
                    two =  [(col + i, row + j), (col + j, row + i)]
                    for p in two:
                        if not self.inBounds(p[0], p[1]) or self.isOccupied(p[0], p[1]) == 2:
                            continue
                        else:
                            out[p[0]][p[1]] = 1

        if piece == 6:          #KING
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    if i == 0 and j == 0 or not self.inBounds(col + i, row + j) or self.isOccupied(col + i, row + j) == 2:
                        continue
                    else:
                        out[col + i][row + j] = 1
        if flip:
            self.turn = not self.turn
        return out 

    def findKing(self):     # find the king of the current player
        pos = 0
        for i in range(8):
                for j in range(8):
                    if self.turn and self.board[i][j] == 6:
                        pos = (i, j)
                        break
                    elif not self.turn and self.board[i][j] == 16:
                        pos = (i, j)
                        break
                if pos:
                    break
        return pos

    def handleCastleFlags(self, coords):    # given that a move is already legal, update flags if piece moved is a rook or king.
        p = self.board[coords[0]][coords[1]] % 10           # this function not going to reduce performance by much, but still messy
        if p != 2 and p != 6: return
        index = 0
        if not self.turn: index = 1
        if p == 6: self.flags[3*index] = True
        if self.heldPiece[0] == 0: self.flags[3*index + 1] = True
        if self.heldPiece[0] == 7: self.flags[3*index + 2] = True

    def enPassantCheck(self, coords):       # if the move as shown by the destination coords is en passant, complete if legal
        p = self.board[self.heldPiece[0]][self.heldPiece[1]]
        if p % 10 != 1: return False
        if self.turn and p > 10: return False       # consider making function for piece/turn agreement
        if not self.turn and p < 10: return False

        if self.turn and not (self.heldPiece[1] == 4 and coords[1] == 5): return False
        if not self.turn and not (self.heldPiece[1] == 3 and coords[1] == 2): return False
        
        newRow = coords[1]
        col = coords[0]
        if (self.inBounds(col, newRow) and self.isOccupied(col, newRow) == 0 and 
        self.lastMove == (col, self.heldPiece[1]) and self.board[col][self.heldPiece[1]] % 10 == 1):
            self.board[col][self.heldPiece[1]] = 0
            self.move(self.heldPiece, coords)
            return True
        return False

    def castleCheck(self, coords):      # same basic idea as enPassantCheck
        p = self.board[self.heldPiece[0]][self.heldPiece[1]]
        if p % 10 != 6:
            return False
        if self.turn and p > 10: return False
        if not self.turn and p < 10: return False
        if self.inCheck(): return False

        index = 0
        if not self.turn: index = 1
        row = [0, 7]
        if coords == (2, row[index]) and not self.flags[3*index] and not self.flags[3*index+1]:
            for i in range(3, 1, -1):
                if self.board[i][row[index]] or (self.inCheck((i, row[index]))): return False
            self.move((0, row[index]), (3, row[index]))
            self.move((4, row[index]), (2, row[index]))
            return True
        elif coords == (6, row[index]) and not self.flags[3*index] and not self.flags[3*index+2]:
            for i in range(5, 7, 1):
                if self.board[i][row[index]] or (self.inCheck((i, row[index]))): return False
            self.move((7, row[index]), (5, row[index]))
            self.move((4, row[index]), (6, row[index]))
            return True
        return False

    def specialMoveCheck(self, coords):
        c, e = self.castleCheck(coords), self.enPassantCheck(coords)
        return (c or e, e)

    def inCheck(self, pos = 0, flip = False):   # is the player whose turn it is currently in check or not? if so, return list of attackers
        o = []
        if not pos:
            pos = self.findKing()
        for i in range(8):
            for j in range(8):
                pieceColorBool = self.board[i][j] > 10
                if flip: pieceColorBool = not pieceColorBool
                if self.turn and pieceColorBool: 
                    if self.availableMoves(i, j, not flip)[pos[0]][pos[1]]:
                        o.append((i, j, self.board[i][j]))
                elif not self.turn and not pieceColorBool: # probably all of this one if statement in the future
                    if self.availableMoves(i, j, not flip)[pos[0]][pos[1]]:
                        o.append((i, j, self.board[i][j]))
        return o

    def moveInCheck(self, start, move):     # return number of attackers if a move will put you in check
        piece = self.board[move[0]][move[1]]
        self.move(start, move)
        out = self.inCheck()
        self.move(move, start)
        self.board[move[0]][move[1]] = piece
        return len(out)

    def isCheckmate(self):      # readability
        i = self.inCheck()
        
        if not len(i): return False

        pos = self.findKing()
        
        for x in range(-1, 2, 1):
            for y in range(-1, 2, 1):
                x1 = pos[0] + x
                y1 = pos[1] + y
                if (x or y) and self.inBounds(x1, y1):
                    if self.turn and self.board[x1][y1] < 10 or not self.turn and self.board[x1][y1] > 10:
                        continue
                    if not self.moveInCheck(pos, (x1, y1)): return False

        if len(i) == 2: return True

        counterAttackers = self.inCheck((i[0][0], i[0][1]), True)
        for ca in counterAttackers:
            if not self.moveInCheck((ca[0], ca[1]), (i[0][0], i[0][1])): return False
        if i[0][2] == 1 or i[0][2] == 3: return True

        xdir, ydir = 0, 0
        if pos[0] > i[0][0]: xdir = -1
        elif pos[0] < i[0][0]: xdir = 1
        if pos[1] > i[0][1]: ydir = -1
        elif pos[1] < i[0][1]: ydir = 1

        x, y = pos[0] + xdir, pos[1] + ydir
        while (x != i[0][0] or y != i[0][1]):
            defenders = self.inCheck((x, y), True)
            for d in defenders:
                if not self.moveInCheck((d[0], d[1]), (x, y)): return False
            x += xdir
            y += ydir

        return True

    def tryMove(self, start, end, promotePiece):
        self.heldPiece = start
        check, capture = False, False
        spm = self.specialMoveCheck(end)
        capture = spm[1]
        if spm[0]:
            self.turn = not self.turn
            if self.inCheck(): check = True
        elif self.availableMoves(start[0], start[1])[end[0]][end[1]]:
            takenPiece = self.board[end[0]][end[1]]
            self.move(start, end)
            if not len(self.inCheck()):
                self.handleCastleFlags(end)
                self.lastMove = end
                if (promotePiece != None):
                    self.promote(promotePiece, end)
                self.turn = not self.turn
                if self.inCheck(): check = True
                if takenPiece: capture = True
            else:
                self.move(end, start)
                self.board[end[0]][end[1]] = takenPiece
                return None
        else: return None
        move = self.storage.pushMove(start, end, promotePiece)
        if self.isOnline and self.onlineTurn:
            cmt = threading.Thread(target=self.client.clientMove, args=(move,), daemon=True)
            cmt.start()
        checkmate = self.isCheckmate() or self.storage.game.is_stalemate()
        self.ply = self.storage.game.ply()
        return [capture, check, checkmate]
