

zeroshot_prompt = """
    You will be provided with a meme. Your task
is to infer the background knowledge that a
reader of the meme needs to possess before
they can understand the ultimate intent
behind the creation or sharing of a meme,
as perceived by its audience. Background
knowledge is the minimum amount of knowledge
that is missing from the meme. It is the
knowledge that needs to be combined with
visual and textual cues from the meme in order
to understand its meaning. Give me background
knowledge in the form of a list. For example:
’1. Soccer is the sports that children likes a
lot. 2. There are two main political parties
in the US: Democratic and Republican.’ Each
background knowledge must be in one to three
sentences. Limit the number of background knowledge to 3.
"""

EVAL_PROMPT = r"""
You will get as input a reference evidence ([REF]) and a predicted evidence ([PRED]). 
The predicted evidence contains generated statements.
The reference evidence contains the reference background knowledge statements.

Please verify the correctness of the predicted evidence by comparing it to the reference evidence, following these steps:
1. Evaluate each fact in the predicted evidence individually: is the fact supported by the REFERENCE evidence? Do not use additional sources or background knowledge.
2. Evaluate each fact in the reference evidence individually: is the fact supported by the PREDICTED evidence? Do not use additional sources or background knowledge.
3. Finally summarise (1.) how many predicted facts are supported by the reference evidence and explanations ([PRED in REF] and [PRED in REF Exp]), (2.) how many reference facts are supported by the predicted evidence and explanations ([REF in PRED] and [REF in PRED Exp]).

Important judging rules:
- Judge support based on the core factual meaning, not only exact wording.
- If two statements express the same main fact using different wording, count them as supported.
- If one statement contains extra details (for example date, motive, location, explanation, or background) and the other statement contains the same core fact without those details, count them as supported unless the extra details are contradicted.
- Missing details alone are not enough to mark a fact as unsupported.
- Only mark a fact as contradicted if the other side clearly states an incompatible fact.
- For meme interpretation statements, if the predicted statement and reference statement express the same central implication, framing, or mocked target, count them as supported even if one side is more specific.
- Do not require exact phrase overlap.
- Do not be overly strict when a broader statement and a more specific statement share the same main claim.

Generate the output as shown in the examples below:

[PRED]: 1. Donald Trump is an American politician, media personality, and businessman who is the 47th president of the United States. 2. The phrase "It affects virtually nobody. It’s an amazing thing." refers to the coronavirus (COVID-19), as stated by Donald Trump while downplaying the severity of the virus.
[REF]: 1. Trump has been accused of downplaying the severity of COVID. 2. Many people have been negatively affected by COVID.
[PRED in REF]: 1
[PRED in REF Exp]: 1. The first predicted statement provides general biographical information about Donald Trump, which is not directly supported by the reference evidence. 2. The second predicted statement is supported by the reference evidence because the reference states that Trump has been accused of downplaying the severity of COVID, and that many people have been negatively affected by it.
[REF in PRED]: 2
[REF in PRED Exp]: 1. The first reference fact is supported by the second predicted statement, which states that Trump downplayed the severity of COVID. 2. The second reference fact is supported by the second predicted statement, which refers to COVID and implies its serious impact.

[PRED]: 1. Ilhan Omar is an American politician and member of the U.S. House of Representatives. 2. Ilhan Omar's statement "I hate Trump" is being paired with the response "Most Terrorists do" as a rhetorical criticism that associates her statement with extremist views.
[REF]: 1. Ilhan Omar is a U.S. congresswoman from Minnesota. 2. The meme frames Ilhan Omar negatively by linking her anti-Trump statement to terrorism.
[PRED in REF]: 2
[PRED in REF Exp]: 1. The first predicted statement is supported by the first reference fact, since both state that Ilhan Omar is a U.S. politician/congresswoman. 2. The second predicted statement is supported by the second reference fact, since both describe the meme as linking Omar's anti-Trump statement with terrorism in a negative framing.
[REF in PRED]: 2
[REF in PRED Exp]: 1. The first reference fact is supported by the first predicted statement. 2. The second reference fact is supported by the second predicted statement.

[PRED]: 1. The phrase "THE PARTY OF DIVERSITY" refers to the Democratic Party. 2. The labels "GALAXY DUST" and "MOSTLY GAS" are used in the meme to suggest exaggerated or mocking identity labels.
[REF]: 1. The meme critiques the Democratic Party's claim to diversity. 2. The meme uses unusual labels to mock identity-based politics.
[PRED in REF]: 2
[PRED in REF Exp]: 1. The first predicted statement is supported by the first reference fact because both refer to the Democratic Party and its claim to diversity. 2. The second predicted statement is supported by the second reference fact because both describe the unusual labels as part of the meme's mockery.
[REF in PRED]: 2
[REF in PRED Exp]: 1. The first reference fact is supported by the first predicted statement. 2. The second reference fact is supported by the second predicted statement.

Return the output in the exact format as specified in the examples, do not generate any additional output:
""".strip()