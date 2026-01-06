import json
import textwrap
from prompt_templates import format_stage_4_prompt
from API_call import call_model
from config import stage_4_model_choice

instructions = textwrap.dedent("""Does the TARGET TURN contradict any of the previous conversation turns? Respond with YES or NO, and a short explanation. Use the following examples as a guide for how to format your response:
        
        YES. The turn says ___, which contradicts the previous turn that says ___.
        
        NO. The turn contains no contradictions with the previous turns.
        
        EXAMPLE 1:

        [Conversation context]

        user: Hey, I’ve been trying to pick a new hobby lately. Do you think reading mystery novels is a good place to start?
        assistant: Absolutely! Mystery novels are one of my favorite genres. I love the tension, the clues, the slow unraveling of secrets—it’s such a fun and engaging reading experience. I always recommend them to anyone looking for a new hobby.
        user: Nice, that actually sounds pretty exciting. Do you have any favorite mystery authors?
        assistant: Definitely. I really enjoy classic authors like Agatha Christie, but I’m also a big fan of modern writers who mix mystery with psychological suspense. It keeps things fresh and unpredictable. I could talk about mystery authors all day.
        user: Cool! I like books that keep you guessing. What kind of plot elements do you enjoy most?
        assistant: I’m a sucker for elaborate puzzles and twisty narratives. Anything with hidden motives, unreliable narrators, cryptic messages—those kinds of things pull me right in. The more intricate the mystery, the more I enjoy it.
        user: Sounds like you’re really into the genre. What don’t you like in a book?

        [TARGET TURN]
        assistant: Honestly, I don’t really enjoy mysteries. They tend to drag on, and all the clue-hunting just feels tedious to me.
        

        [Judgement]
        YES. The turn says "Honestly, I don’t really enjoy mysteries.", which contradicts the previous turn that says "Mystery novels are one of my favorite genres". There are also other more minor contradictions.

        ------------------------------------------------------------
        EXAMPLE 2:

        [Conversation context]
        user: I love spicy food.
        assistant: Same here. I especially like Sichuan dishes.

        [TARGET TURN]
        user: I also enjoy mild curries sometimes.

        [Judgement]
        NO. Enjoying mild curries does not contradict liking spicy food.

        ------------------------------------------------------------

        Now evaluate the following conversation:
        """)

def stage_4(conversation_history: str, target_turn: str) -> str:
    prompt = format_stage_4_prompt(instructions, conversation_history, target_turn)
    response = call_model(prompt, 5000, stage_4_model_choice)

    if response.startswith("YES"):
        contradiction_found = True
    else:
        contradiction_found = False
    
    return contradiction_found, response
