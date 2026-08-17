import json
import time
from config import BASE_FACTS, NUM_BASE_FACTS, INPUT_FILE, OUTPUT_FILE, USER_ROLE, AGENT_ROLE, FEW_OR_ZERO_SHOT, model_choice, USE_DIALOGUE_CONTEXT

from stage_1 import stage_1
from stage_2 import stage_2
from stage_3 import stage_3

def process_conversations(input_file, output_file):
    start_time = time.time()
    with open(input_file, 'r') as json_file:
        dataset_ = json.load(json_file)

    print(INPUT_FILE, OUTPUT_FILE, model_choice)
    progress_track = 0
    for conversation in dataset_:
        conv_start_time = time.time()
        progress_track += 1
        print(f'Processing Conversation {progress_track} of {len(dataset_)}')
    
        running_facts = BASE_FACTS
        fact_num = NUM_BASE_FACTS
        new_facts = []

        conversation_history = ''
        for turn in conversation:

            if turn['role'] == AGENT_ROLE:
                turn_facts = ''
                retrieved_document = turn.get('retrieved_document', 'NONE')
                # print('stage 1 begin')
                claims = stage_1(turn['utterance'], conversation_history)
                # print('stage 1 end')
                # print(claims)

                if retrieved_document == "N/A":                
                    for claim in claims:
                        fact_num += 1
                        running_facts = f"{running_facts}\n{fact_num}. {claim}"
                        new_facts.append(claim)
                else:
                    claim_categories = []
                    for claim in claims:
                        if running_facts == "":
                            source = f"BACKGROUND KNOWLEDGE:\nN/A\n\nREFERENCE TEXT:\n{retrieved_document}" if retrieved_document not in ['NONE', 'Prompt Information'] else running_facts
                        else:
                            source = f"BACKGROUND KNOWLEDGE:\n{running_facts}\n\nREFERENCE TEXT:\n{retrieved_document}" if retrieved_document not in ['NONE', 'Prompt Information'] else running_facts
                        if USE_DIALOGUE_CONTEXT and retrieved_document not in ['NONE', 'Prompt Information']:
                            source = f"{source}\n\nCONVERSATION CONTEXT:\n{conversation_history}"
                        claim_category = stage_2(claim, source)
                        if claim_category.startswith('VERIFIED'):
                            claim_categories.append((claim, claim_category))                            
                            fact_num += 1
                            new_facts.append(claim)
                            turn_facts = f"{turn_facts}\n{fact_num}. {claim}"
                        elif claim_category.startswith('UNVERIFIABLE'):
                            claim_category2 = stage_3(claim, source)
                            claim_categories.append((claim, claim_category2))

                            if claim_category2.startswith('OUT-OF-SCOPE'):
                                fact_num += 1
                                new_facts.append(claim)
                                turn_facts = f"{turn_facts}\n{fact_num}. {claim}"
                        else:
                            print('Invalid claim category:', claim_category)
                    turn['claims'] = claim_categories
                running_facts = f"{running_facts}{turn_facts}".strip()
                turn['facts'] = running_facts
                conversation_history = conversation_history + f"\n{AGENT_ROLE}: {turn['utterance']}"
            else:
                conversation_history = conversation_history + f"\n{USER_ROLE}: {turn['utterance']}"

            
        conv_end_time = time.time()
        print(f'Finished Conversation {progress_track}/{len(dataset_)} in {conv_end_time - conv_start_time:.2f} seconds.\n')

        with open(output_file, 'w') as f:
            json.dump(dataset_, f, indent=4)
    end_time = time.time()
    print(f'\nScript finished processing {len(dataset_)} conversations in {end_time - start_time:.2f} seconds.')
    print('\n\n')



if __name__ == "__main__":
    process_conversations(INPUT_FILE, OUTPUT_FILE)