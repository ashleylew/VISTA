from API_call import call_model
import textwrap
import json
from config import USER_ROLE, AGENT_ROLE, FEW_OR_ZERO_SHOT, model_choice
from prompt_templates import format_prompt

instructions = textwrap.dedent(f"""\
        Extract all distinct factual or belief-based statements from the TARGET TURN only.
        
        - Break down compound or complex sentences into individual, atomic claims.
        - Resolve any references (e.g., use proper names instead of "he", "it", etc.).
        - Capture each idea as stated or implied by the speaker, regardless of accuracy.
        - Treat presuppositions as mandatory outputs. If an utterance contains a presupposition, you MUST list it as one or more separate atomic statements. For example, if the utterance is "{AGENT_ROLE}: I didn't know that embroidery is a needlework technique", you MUST list "Embroidery is a needlework technique" as an atomic statement as well as "The {AGENT_ROLE} didn't know that embroidery is a needlework technique".
        - Do NOT evaluate or comment on whether a statement is correct.
        - Simply list the statements as they appear to be intended.

        Format your output as a numbered list of standalone claims.
        Do not include any commentary or qualification.
        Only include what was communicated in the TARGET TURN.""")


STAGE_1_FEW_SHOT_EXAMPLES = {
    1: {
        "input": 
            textwrap.dedent(f'''\
            CONVERSATION HISTORY:
            {USER_ROLE}: Hi Dr. Ilsa, what can I expect at the COSI museum?
            {AGENT_ROLE}: Welcome! At COSI, you can explore live science shows, special exhibits, and other activities that make science fascinating and fun. You'll definitely enjoy your trip!
            {USER_ROLE}: Are there any live science shows happening right now?
            {AGENT_ROLE}: Absolutely! You can check the program schedule at the Box Office or Guest Services Desk for details about the current live shows.
            {USER_ROLE}: I heard there was an outdoor exhibit?

            TARGET TURN:
            {AGENT_ROLE}: Big Science Park is an outdoor laboratory full of exciting science activities. You can try lifting a car with a lever or even roll a giant granite sphere!'''),
        
        "output": textwrap.dedent("""\
                1. Big Science Park is an outdoor laboratory.
                2. Big Science Park features exciting science activities.
                3. Visitors can try lifting a car using a lever at Big Science Park.
                4. Visitors can roll a giant granite sphere at Big Science Park.""")
        },

    2: {
        "input": textwrap.dedent(f'''\
                CONVERSATION HISTORY:
                {USER_ROLE}: Hi! What's your name?
                {AGENT_ROLE}: My name is Dr. Ilsa. I'm named after the woman who founded the OSU Linguistics department, Dr. Ilsa Lehiste. What's your name?
                {USER_ROLE}: My name is Marie!

                TARGET TURN:
                {AGENT_ROLE}: Hello Marie, welcome to COSI.'''),

        "output": textwrap.dedent(f"""\
                1. The {USER_ROLE}'s name is Marie.
                2. The {USER_ROLE} and {AGENT_ROLE} are at COSI.""")
        },
        
    3: {
        "input": textwrap.dedent(f'''\
                CONVERSATION HISTORY:
                {USER_ROLE}: Hi Dr. Ilsa. Where did your name come from?

                TARGET TURN:
                {AGENT_ROLE}: I'm named after Dr. Ilse Lehiste who founded the Department of Linguistics at the Ohio State University.'''),

        "output": textwrap.dedent(f"""\
                1. The {AGENT_ROLE} is named after Dr. Ilse Lehiste.
                2. Ilse Lehiste founded the Department of Linguistics at the Ohio State University.""")
        },
    4: {
        "input": textwrap.dedent(f'''\
                CONVERSATION HISTORY:
                {USER_ROLE}: Hi! What is this place?

                TARGET TURN:
                {AGENT_ROLE}: Hello! The Language Pod is a research lab from Ohio State University. Here we study different aspects of language like how people talk differently, how children learn language, and how computers and humans can interact using language.'''),
                
        "output": textwrap.dedent(f"""\
                1. The Language Pod is a research lab.
                2. The Language Pod is from Ohio State University.
                3. The Language Pod studies different aspects of language.
                4. One focus of the Language Pod is how people talk differently.
                5. Another focus of the Language Pod is how children learn language.
                6. The Language Pod also studies how computers and humans can interact using language.""")
        },

    5: {
        "input": textwrap.dedent(f'''\
                CONVERSATION HISTORY:
                {USER_ROLE}: Could you recommend a few Pixar movies, please?
                {AGENT_ROLE}: Have you seen A Bug’s Life? That was one of their earlier films.
                {USER_ROLE}: I haven’t! Is that one of their more popular ones?
                {AGENT_ROLE}: It came out in 1998, right after Toy Story. Would you like me to suggest a couple more Pixar films?
                {USER_ROLE}: Yes, please. Maybe one or two more, just so I have some options.

                TARGET TURN:
                {AGENT_ROLE}: I'll give you my personal favorite, Finding Nemo. Do you know that one?'''),

        "output": textwrap.dedent(f"""\
                1. The {AGENT_ROLE} is giving the {USER_ROLE} a recommendation for a Pixar movie.
                2. The {AGENT_ROLE} recommends the movie Finding Nemo.
                3. Finding Nemo is a Pixar movie.
                4. Finding Nemo is the {AGENT_ROLE}'s personal favorite Pixar movie.""")
    },
    6: {
        "input": textwrap.dedent(f'''\
            CONVERSATION HISTORY:
            {USER_ROLE}: I love superhero movies.
            {AGENT_ROLE}: Me too. I'm a big fan of Iron Man.
            {USER_ROLE}: Yeah Robert Downey Jr. is a favorite.

            TARGET TURN:
            {AGENT_ROLE}: I didn't know that RDJ was in that movie.'''),
        "output": textwrap.dedent(f"""\
            1. Iron Man is a movie.
            2. Robert Downey Jr. was in Iron Man.
            3. The {AGENT_ROLE} did not know that Robert Downey Jr. was in Iron Man.""")
        }
}

def stage_1(turn, conversation_history):
    prompt = format_prompt(FEW_OR_ZERO_SHOT, instructions, STAGE_1_FEW_SHOT_EXAMPLES, conversation_history, turn)
    response = call_model(prompt, 5000, model_choice)

    try:
        return [claim[3:].rstrip() for claim in response.split('\n') if claim]
    except:
        print('Invalid response format:')
        print(response)
        print('END OF ERROR')
        return []