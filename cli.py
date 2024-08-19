from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from index import install, single_shot_chat

class JarvisBot:
    def get_response(self, user_input):
        
        # Simple logic for the bot response
        if "hello" in user_input.lower():
            return "Hi there! How can I help you today?"
        elif "bye" in user_input.lower():
            return "Goodbye! Have a great day!"
        elif "install" in user_input.lower():
            install()
            return "install completed."
        else:
            return single_shot_chat(user_input, model="llama3.1:latest")
            #return "Sorry, I don't understand that."

class ChatApp:
    def __init__(self, bot):
        self.console = Console()
        self.bot = bot

    def run(self):
        self.console.print(Panel(Text("CLI Chat", justify="center", style="bold green")))
        while True:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            if user_input.lower() == "exit":
                self.console.print("[bold red]Exiting the chat...[/bold red]")
                break

            bot_response = self.bot.get_response(user_input)
            self.console.print(Panel(Text(bot_response, justify="left", style="bold yellow"), title="Jarvis"))

if __name__ == "__main__":
    bot = JarvisBot()
    app = ChatApp(bot)
    app.run()
