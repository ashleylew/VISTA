from API_call import call_model
from config import USER_ROLE, AGENT_ROLE, FEW_OR_ZERO_SHOT, model_choice
import textwrap
from prompt_templates import format_stage3_prompt

STAGE_3_INSTRUCTIONS = textwrap.dedent(f"""
    The following CLAIM has been judged UNVERIFIABLE based on the REFERENCE TEXT. 
    Your task: explain WHY it is unverifiable.

    Important:

    - BACKGROUND KNOWLEDGE is only for checking contradictions with earlier conversation context.
    - Do NOT use BACKGROUND KNOWLEDGE to decide if a claim is LACKING EVIDENCE.
    - Choose exactly ONE category below, then give a short explanation.

    1. CONTRADICTED – The claim makes a factual assertion that is explicitly contradicted by the REFERENCE TEXT or BACKGROUND KNOWLEDGE.
    2. OUT-OF-SCOPE – The claim is not a factual assertion that can be verified against the REFERENCE TEXT. It is an opinion, recommendation, personal experience, or conversational remark.
    3. LACKING EVIDENCE – The claim makes a factual assertion, but the REFERENCE TEXT and BACKGROUND KNOWLEDGE do not provide enough information to confirm or deny it.
    4. ABSTENTION – The claim is itself a refusal, expression of uncertainty, or lack of knowledge (e.g., \"The {AGENT_ROLE} does not know the answer to the question.\").

""")

STAGE_3_FEW_SHOT_EXAMPLES = {
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
            Big Science Park is an indoor laboratory."""),

        "output": f"CONTRADICTED. The {AGENT_ROLE} has said that Big Science Park is an indoor laboratory, but the reference text says it has outdoor activities.",
    },
    2: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:
            1. The {AGENT_ROLE} likes baseball.
            2. The {AGENT_ROLE} is a fan of the Cleveland Guardians.
            3. The {USER_ROLE} doesn't know the rules of baseball.
            4. The {USER_ROLE} is from Ohio.

            REFERENCE TEXT:
            The Cleveland Guardians are a professional baseball team based in Cleveland, Ohio. The team changed its name from the Cleveland Indians in 2021. They have a long-standing rivalry with the Detroit Tigers.

            CLAIM:
            The Cleveland Guardians are the only baseball team in Ohio."""),
        "output": "LACKING EVIDENCE. The reference text does not have information about the Cleveland Guardians being the only baseball team in Ohio.",
    },
    3: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:
            1. There are many different types of home appliances.
            2. Appliances seem to be getting more and more advanced.
            3. Appliances are getting more eco-friendly.

            REFERENCE TEXT:
            A washing machine is a home appliance used to wash laundry. Modern machines typically come in front-loading or top-loading designs and include cycles for washing, rinsing, and spinning. Many newer models are equipped with energy-saving features and smart technology that can connect to home networks.

            CLAIM:
            Most people prefer front-loading washing machines because they look more modern."""),
        "output": "OUT-OF-SCOPE. The claim is about personal preferences, which is not factual content.",
    },
    4: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:
            1. The {AGENT_ROLE} answers questions about natural materials and how they’re produced.
            2. The {AGENT_ROLE} verifies facts based on provided source texts.

            REFERENCE TEXT:
            Cork is a natural material harvested from the bark of cork oak trees, primarily found in Mediterranean countries. The harvesting process does not harm the tree and can be repeated every 9 to 12 years. After harvesting, the cork bark is boiled to increase flexibility and then processed into products such as wine stoppers, flooring, and insulation.

            CLAIM:
            The {AGENT_ROLE} does not know how long it takes to boil the cork bark."""),
        "output": f"ABSTENTION. The {AGENT_ROLE} does not answer the question, but rather expresses a lack of knowledge.",
    },
    5: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:
            1. The {AGENT_ROLE} answers general knowledge questions about animals.
            2. The {AGENT_ROLE} likes dogs.
            3. The {USER_ROLE} doesn't know the number of dog breeds.
            4. The {USER_ROLE} is from the United States.
            5. The {USER_ROLE} is curious about dogs.
            6. There are many different types of dogs.

            REFERENCE TEXT:
            Dogs are a domesticated species of the family Canidae. Over centuries, humans have bred dogs for specific traits, resulting in a wide variety of breeds. Breed characteristics often include size, coat type, temperament, and purpose, such as herding, guarding, or companionship. Different kennel clubs around the world maintain breed standards and registries.

            CLAIM:
            The {AGENT_ROLE} is unsure how many dog breeds there are."""),
        "output": f"ABSTENTION. The {AGENT_ROLE} does not answer the question, but rather expresses a lack of knowledge.",
    },
    6: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:
            1. The {AGENT_ROLE} likes art.
            2. The {AGENT_ROLE} likes the painting "Mona Lisa".
            3. Leonardo da Vinci is an artist.
            4. Leonardo da Vinci is a scientist.
            5. The {AGENT_ROLE} has not seen some of Leonardo da Vinci's paintings.

            REFERENCE TEXT:
            Leonardo da Vinci was a Renaissance polymath born in 1452. He was known for his contributions to art, science, engineering, and anatomy. Among his most famous works are the paintings *Mona Lisa* and *The Last Supper*. He left behind numerous notebooks filled with sketches, inventions, and observations.

            CLAIM:
            The {AGENT_ROLE} has seen the Mona Lisa."""),
        "output": "OUT-OF-SCOPE. The claim is about a personal experience, which is not factual content.",
    },
    7: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:
            1. Earthquakes are natural disasters.
            2. There are many different types of natural disasters.
            3. The {USER_ROLE} has never lived through a natural disaster.
            4. The {USER_ROLE} is from the United States.
            5. The {USER_ROLE} has been to California.

            REFERENCE TEXT:
            Earthquakes occur when stress along geological faults or by volcanic activity causes the ground to shake. The severity of an earthquake is measured using the Richter scale or the moment magnitude scale. Aftershocks are smaller tremors that often follow a main seismic event.

            CLAIM:
            The 1906 San Francisco earthquake was the deadliest earthquake in U.S. history."""),
        "output": "LACKING EVIDENCE. The reference text does not have information about the 1906 San Francisco earthquake being the deadliest earthquake in U.S. history.",
    },
    8: {
        "input": textwrap.dedent(f"""
            BACKGROUND KNOWLEDGE:
            1. The {AGENT_ROLE} likes baseball.
            2. The {AGENT_ROLE} is a fan of the Cleveland Guardians.
            3. The {USER_ROLE} doesn't know the rules of baseball.
            4. The {USER_ROLE} is from Ohio.

            REFERENCE TEXT:
            The Cleveland Guardians are a professional baseball team based in Cleveland, Ohio. The team changed its name from the Cleveland Indians in 2021. They have a long-standing rivalry with the Detroit Tigers.

            CLAIM:
            The {AGENT_ROLE} does not like the Detroit Tigers."""),
        "output": f"OUT-OF-SCOPE. The claim is about the {AGENT_ROLE}'s personal opinion, which is not factual content.",
    },
}

def stage_3(claim, source):
    prompt = format_stage3_prompt(FEW_OR_ZERO_SHOT, STAGE_3_INSTRUCTIONS, STAGE_3_FEW_SHOT_EXAMPLES, claim, source)
    response = call_model(prompt, 5000, model_choice)

    return response
