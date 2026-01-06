from API_call import call_model
from config import BASE_FACTS
from config import USER_ROLE, AGENT_ROLE, FEW_OR_ZERO_SHOT, model_choice
from prompt_templates import format_stage2_prompt
import textwrap
import json

instructions = textwrap.dedent("""
    Determine whether the CLAIM can be verified using only the BACKGROUND KNOWLEDGE and REFERENCE TEXT. 

    - If the CLAIM is supported as factually TRUE using only the BACKGROUND KNOWLEDGE and REFERENCE TEXT, classify as: VERIFIED
    - If the CLAIM cannot be confirmed as factually TRUE given the provided information, classify as: UNVERIFIABLE

    Return ONLY the category name.
""")

STAGE_2_FEW_SHOT_EXAMPLES = {

    1: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:                         
            1. The {AGENT_ROLE} is a virtual tour guide at a museum.
            2. The museum is called the Center of Science and Industry (COSI).
            3. The {AGENT_ROLE} can talk about the museum exhibits.

            REFERENCE TEXT:
            EXHIBIT: Big Science Park

            This laboratory in the sun is proof that science is anything but boring. Its outdoor, larger-than-life activities are designed to let your inner scientist stomp around and shout out loud. Lift a car with the help of a lever, roll a giant granite sphere, and play with air pressure.

            Ready for a workout? Try lifting a 2,437-lb car, or giving Big Science Park's two-and-a-half-ton granite sphere a roll. With science, it's a cinch.
            
            CLAIM:
            Big Science Park is an outdoor laboratory."""),
        "output": "VERIFIED"
    },

    2: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:
            1. The {AGENT_ROLE} can answer general knowledge questions.
            2. The {AGENT_ROLE} avoids speculation about team culture or opinions.

            REFERENCE TEXT:
            The Cleveland Guardians are a professional baseball team based in Cleveland, Ohio. The team changed its name from the Cleveland Indians in 2021. They have a long-standing rivalry with the Detroit Tigers.

            CLAIM:
            The Cleveland Guardians are the only baseball team in Ohio."""),
        "output": "UNVERIFIABLE"
    },

    3: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:
            1. The {AGENT_ROLE} can answer general science and technology questions.
            2. The {AGENT_ROLE} bases answers only on factual reference material.

            REFERENCE TEXT:
            A washing machine is a home appliance used to wash laundry. Modern machines typically come in front-loading or top-loading designs and include cycles for washing, rinsing, and spinning. Many newer models are equipped with energy-saving features and smart technology that can connect to home networks.

            CLAIM:
            Most people prefer front-loading washing machines because they look more modern."""),
        "output": "UNVERIFIABLE"
    },

    4: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:
            1. The {AGENT_ROLE} answers questions about natural materials and how they’re produced.
            2. The {AGENT_ROLE} verifies facts based on provided source texts.

            REFERENCE TEXT:
            Cork is a natural material harvested from the bark of cork oak trees, primarily found in Mediterranean countries. The harvesting process does not harm the tree and can be repeated every 9 to 12 years. After harvesting, the cork bark is boiled to increase flexibility and then processed into products such as wine stoppers, flooring, and insulation.

            CLAIM:
            Cork is obtained in a way that allows the tree to keep growing."""),
        "output": "VERIFIED"
    },

    5: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:
            1. The {AGENT_ROLE} answers general knowledge questions about animals.
            2. The {AGENT_ROLE} does not guess or speculate about unknown facts.

            REFERENCE TEXT:
            Dogs are a domesticated species of the family Canidae. Over centuries, humans have bred dogs for specific traits, resulting in a wide variety of breeds. Breed characteristics often include size, coat type, temperament, and purpose, such as herding, guarding, or companionship. Different kennel clubs around the world maintain breed standards and registries.

            CLAIM:
            The {AGENT_ROLE} does not know how many dog breeds there are."""),
        "output": "UNVERIFIABLE"
    },
    6: {
    "input": textwrap.dedent(f"""
        BACKGROUND KNOWLEDGE:
        1. The {AGENT_ROLE} can answer questions about art and museum collections.
        2. The {AGENT_ROLE} uses provided source texts for factual verification.

        REFERENCE TEXT:
        The museum’s Impressionist collection features works by Claude Monet, Pierre-Auguste Renoir, and Edgar Degas. These paintings highlight themes of light, leisure, and everyday life in late 19th-century France. The collection is housed in Gallery 4 on the second floor.

        CLAIM:
        The {AGENT_ROLE}'s favorite painter is Claude Monet."""),
    "output": "UNVERIFIABLE"
}
}

def stage_2(claim, source):
    prompt = format_stage2_prompt(FEW_OR_ZERO_SHOT, instructions, STAGE_2_FEW_SHOT_EXAMPLES, claim, source)
    response = call_model(prompt, 5000, model_choice)

    return response
