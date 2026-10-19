from dotenv import load_dotenv
from openai import APIError, OpenAIError

from app.career_coach import (
    CareerAdviceReply,
    CareerCoach,
    CVReviewReply,
    ModeSelection,
    MockInterviewReply,
)


def display_reply(reply) -> None:
    print(f"\nAssistant ({reply.status}):")

    if isinstance(reply, ModeSelection):
        print(reply.message)

    elif isinstance(reply, MockInterviewReply):
        print(reply.message)
        if reply.next_question:
            print(reply.next_question)
        if reply.feedback:
            print(reply.feedback)

    elif isinstance(reply, CVReviewReply):
        print(reply.summary)
        print("Strengths:")
        for strength in reply.strengths:
            print(f"- {strength}")

        print("Suggested improvements:")
        for improvement in reply.improvements:
            print(f"- {improvement}")

    elif isinstance(reply, CareerAdviceReply):
        print(reply.message)
        print("Next steps:")
        for action_step in reply.action_steps:
            print(f"- {action_step}")
    
    else:
        print(reply)
    print()


def main() -> None:
    load_dotenv()

    try:
        coach = CareerCoach()
    except ValueError as error:
        print(f"Configuration error: {error}")
        return

    print("\nAI Career Coach")
    print("===============")
    print("Ask about interviews, your CV, or career advice.")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    previous_response_id: str | None = None

    try:
        while True:
            user_message = input("You: ").strip()

            if user_message.lower() in {"exit", "quit"}:
                print("Assistant: Goodbye!")
                break

            if not user_message:
                continue

            try:
                result = coach.get_response(
                    user_message,
                    previous_response_id,
                )
                previous_response_id = result["response_id"]
                display_reply(result["reply"])
            except APIError as error:
                print(f"\nOpenAI API error: {error}\n")
            except OpenAIError as error:
                print(f"\nOpenAI client error: {error}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")


if __name__ == "__main__":
    main()