

# Use OOP to create a game class that stores game state


class HangmanGame():

    def __init__(self, word: str = "hangman", missesToFailure: int = 5):
        self.resetGameState(word, missesToFailure)
        return

    def resetGameState(self, word: str, missesToFailure: int):

        assert len(word.strip().split()) == 1, "Word must not contain any spaces!"

        gameLetters = [c for c in word if c.isalpha() and c.isascii()]
        gameWord = ''.join(gameLetters)
        currentBoard = ["_"] * len(gameLetters) 

        # External, can be shared with the person playing the game
        self.missesToFailure = missesToFailure
        self.gameLetters = gameLetters
        self.gameWord = gameWord
        self.numGuesses = 0
        self.numMisses = 0
        self.gameComplete = False
        self.gameWon = False
        self.currentBoard = currentBoard
        self.lettersGuessed = []
        return


    def getBoardState(self) -> dict:
        return {
            "gameLetters": self.gameLetters,
            "gameWord": self.gameWord,
            "numGuesses": self.numGuesses,
            "numMisses": self.numMisses,
            "gameComplete": self.gameComplete,
            "gameWon": self.gameWon,
            "currentBoard": self.currentBoard,
            "lettersGuessed": self.lettersGuessed
        }

    def isGameComplete(self) -> bool:
        return self.gameWon



    def _assess_win(self):
        # If board is complete and if the board is done, we are cone
        if self.currentBoard == self.gameLetters:
            self.gameWon = True
            self.gameComplete = True 
        elif self.missesToFailure == self.numMisses:
            print("reached num misses")
            self.gameWon = False
            self.gameComplete = True
        else:
            self.gameComplete = False
            self.gameWon = False
        return


    def guessLetter(self, letter: str) -> dict:

        assert len(letter) == 1, "Guess must be a letter!"
        assert self.gameComplete == False, "Cannot guess a letter on a completed game!"

        # If a letter was already guessed, don't count it
        # as a miss and give the player a freebie
        assert letter not in self.lettersGuessed, "Already guessed that letter!"

        # Always increase the number of guesses
        self.numGuesses += 1
        self.lettersGuessed.append(letter)
        
        # Determine if there are any pieces to be filled
        if letter in self.gameLetters:
            # The beef
            self.currentBoard = [(c if c in self.lettersGuessed else "_") for c in self.gameLetters]
        else:
            self.numMisses += 1


        self._assess_win()
        return self.getBoardState()


def test_barely_won():
    print("Test barely winning a game")
    game = HangmanGame("letter", missesToFailure=3)
    game.guessLetter("e")
    game.guessLetter("t")
    game.guessLetter("l")
    game.guessLetter("a")
    game.guessLetter("b")
    game.guessLetter("r")
    assert game.currentBoard == ["l", "e", "t", "t", "e", "r"]
    assert game.numGuesses == 6
    assert game.numMisses == 2
    assert game.gameComplete == True
    assert game.gameComplete == True



def test_won_game():
    print("Test winning a game")
    game = HangmanGame("letter")
    game.guessLetter("e")
    game.guessLetter("t")
    game.guessLetter("l")
    game.guessLetter("r")
    assert game.currentBoard == ["l", "e", "t", "t", "e", "r"]
    assert game.numGuesses == 4
    assert game.numMisses == 0
    assert game.gameComplete == True
    assert game.gameComplete == True


def test_repeated_letter_guess():
    print("Test guessing a repeated letter")
    game = HangmanGame("letter")
    game.guessLetter("e")
    game.guessLetter("t")
    assert game.currentBoard == ["_", "e", "t", "t", "e", "_"]
    assert game.numGuesses == 2
    assert game.numMisses == 0
    assert game.gameComplete == False
    assert game.gameComplete == False


def test_lost_game():
    print("Test losing a game")
    game = HangmanGame("word", missesToFailure=2)
    game.guessLetter("a")
    game.guessLetter("b")
    assert game.gameComplete == True
    assert game.gameWon == False
    try:
        game.guessLetter("c")
    except Exception as e:
        assert str(e) == "Cannot guess a letter on a completed game!"



def test_letter_guessed_twice():
    print("Test guessing a letter twice")
    game = HangmanGame("word")
    game.guessLetter("d")
    try:
        game.guessLetter("d")
    except Exception as e:
        assert str(e) == "Already guessed that letter!"


def test_letter_guessed_incorrectly():
    print("Test guessing an incorrect letter")
    game = HangmanGame("word")
    game.guessLetter("a")
    assert game.currentBoard == ["_", "_", "_", "_"]
    assert game.numGuesses == 1
    assert game.numMisses == 1
    assert game.gameComplete == False
    assert game.gameWon == False


def test_letter_guessed():
    print("Test guessing a letter")
    game = HangmanGame("word")
    game = HangmanGame("letter")
    game.guessLetter("e")
    game.guessLetter("t")
    game.guessLetter("l")
    game.guessLetter("r")
    assert game.currentBoard == ["l", "e", "t", "t", "e", "r"]
    assert game.numGuesses == 4
    assert game.numMisses == 0
    assert game.gameComplete == True
    assert game.gameComplete == True


def test_failed_guess():
    print("Testing failed guess")
    try:
        game = HangmanGame("word")
        game.guessLetter("a b")
    except Exception as e:
        assert str(e) == "Guess must be a letter!"



def test_failed_word():
    print("Testing failed word")
    try:
        game = HangmanGame("bad word")
    except Exception as e:
        assert str(e) == "Word must not contain any spaces!"


def test_initialization():
    print("Testing init")
    game = HangmanGame("myword")
    assert game.gameLetters == ["m", "y", "w", "o", "r", "d"]
    assert game.currentBoard == ["_"] * 6




def runTests():
    test_initialization()
    test_failed_word()
    test_failed_guess()
    test_letter_guessed()
    test_letter_guessed_incorrectly()
    test_letter_guessed_twice()
    test_lost_game()
    test_repeated_letter_guess()
    test_won_game()
    test_barely_won()


    return




if __name__ == "__main__":
    runTests()

    print("======================")
    print("    HANGMAN      ")
    print("======================")
    gameWord = input("Configure game word: ")
    totalMisses = input("Total missues before losing the game: ")
    print("===================")
    print("  let's play! ")
    g = HangmanGame(word=gameWord, missesToFailure=int(totalMisses))
    while g.gameComplete == False:
        guess = input("Guess a letter: ")
        g.guessLetter(guess)
        print(" [ ", g.currentBoard, " ] ", " , misses: ", g.numMisses, "/", g.missesToFailure)
    if g.gameWon:
        print("YOU WON THE GAME!")
    else:
        print("BETTER LUCK NEXT TIME!")
