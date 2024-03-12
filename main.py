from Games.SimpleDrawGame.SimpleDrawGame import SimpleDrawGame
from Games.QuickDrawGame.QuickDrawGame import QuickDrawGame
from colorama import Fore


def main():

    while True:
        print(Fore.YELLOW, "Please select one of the games:")
        print("1. Simple Draw Game")
        print("2. Quick Draw Game")
        print("3. Exit")
        ans = input("Enter game number:")
        try:
            num = int(ans)
        except:
            print(Fore.RED, "Please enter a valid value")
            continue

        if num == 1:
            draw_game = SimpleDrawGame()
            draw_game.run()
        elif num == 2:
            quickdraw_game = QuickDrawGame()
            quickdraw_game.run()
        elif num == 3:
            print("Goodbye!")
            break
        else:
            print(Fore.RED, "Please enter a valid value")



if __name__ == '__main__':
    main()
