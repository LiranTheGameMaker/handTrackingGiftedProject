print("Welcome to the Game Menu!")
print("Please choose a game:")
print("1. Simple Draw Game")
print("2. Simple Maze Game")

# Get user input
choice = input("Enter the number of your choice: ")

# Validate user input
if choice.isdigit():
    choice = int(choice)
    if choice == 1:
        # Import and run Simple Draw Game
        from Games.SimpleDrawGame.SimpleDrawGame import SimpleDrawGame
        SimpleDrawGame()
    elif choice == 2:
        # Import and run Simple Maze Game
        from Games.SimpleMazeGame.SimpleMazeGame import run_maze_game
        run_maze_game()
    else:
        print("Invalid choice. Please choose either 1 or 2.")
else:
    print("Invalid input. Please enter a number.")