#! /usr/bin/env python3
import textwrap

from theoros.guardrails import PromptScanner, OutputScanner
from theoros.theoros_main import theoros_agent

prompt_scanner = PromptScanner()
output_scanner = OutputScanner()

def main():
    running = True
    chat_history = []
    while running:
        query = input("What would you like to query? ('quit' to exit): ")
        if query.lower() == "quit":
            running = False
        elif query.lower() == "history":
            print("Chat history:")
            for i, message in enumerate(chat_history):
                print(f"{i} {message}")
        elif query.lower() == "clear":
            chat_history.clear()
            print("Chat history cleared.")
        elif query.lower() == "help":
            print("Available commands:")
            print("  'quit' - Exit the program")
            print("  'history' - Show chat history")
            print("  'clear' - Clear chat history")
            print("  'help' - Show this help message")
        elif query.lower() == "":
            continue
        else:
            sanitized_prompt, results_valid, results_score = \
                prompt_scanner.scan_input(query)
            bad_input = False
            for scanner_name, scanner_result in results_valid.items():
                if not scanner_result:
                    print(f"Input scanner fail: '{scanner_name}', {results_score[scanner_name]}")
                    bad_input = True
            if bad_input:
                continue

            result = theoros_agent.run_sync(sanitized_prompt, message_history=chat_history)
            chat_history.extend(result.all_messages())
            sanitized_output, results_valid, results_score = output_scanner.scan_output(sanitized_prompt, result.output)
            bad_output = False
            for scanner_name, scanner_result in results_valid.items():
                if not scanner_result:
                    print(f"Output scanner fail: '{scanner_name}', {results_score[scanner_name]}")
                    bad_output = True
            if bad_output:
                continue

            content = result.output
            lines = content.splitlines()
            for line in lines:
                if len(line) > 100:
                    for wrapped_line in textwrap.wrap(line, width=100):
                        print(wrapped_line)
                else:
                    print(line)

if __name__ == '__main__':
    main()