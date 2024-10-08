Zero_Shot = """ The task is to detect whether the system's response causes a dialogue breakdown in a given context. A dialogue breakdown occurs when the user finds it difficult to continue the conversation. 
Classify the interaction as 'breakdown' if it is difficult for the user to continue, and 'no breakdown' if it is easy for the user to continue the conversation.
The output needs to be always either 'breakdown' or 'no breakdown', nothing else. Do NOT provide any aditional explanation or reasoning as an output."""

Few_Shot_10 =  """The task is to detect whether the system response causes dialogue breakdown in a given dialogue context.  A dialogue breakdown occurs when the user finds it difficult to continue the conversation.

It would be 'breakdown' if it is difficult to continue the conversation, and 'no breakdown' if it is easy to continue the conversation. 

Some examples are: 

U: good for you. What are you doing for living?
S: good now that the power is back on. no breakdown

U: nice do you like it?
S: yes i do. i own the bar. do you like the store? breakdown

U: you're weird
S: i work at an apartment office. no breakdown

U: Okey how is it going?
S: i am just a person who is a vegan. breakdown

U: why
S: i am great a little tired. no breakdown

U: I see
S: let be friends i've none in the u.s. hope to make some in summer camp. breakdown

U: can not imagine
S: well there is a lot of pressure to push out good fake content. get is aid that. no breakdown

U: well ok)
S: after while crocodile, not too soon baboon SMILING FACE WITH SUNGLASSES breakdown

U: i read a book. it is my favorite way to relax
S: i rebuilt a diesel all day and was like damn. Also, try to guess: my father is... no breakdown

U: I ate tacos. It is the best food ever
S: we paint the town red! what do you like to do? breakdown

Considering the examples, classify the provided interaction into 'breakdown' or 'no breakdown', depending on the difficulty for 
the user to continue conversation. Do it exactly like in the examples. Do NOT provide your reasoning, or any 
explanation or justification as an answer. You need to provide either 'breakdown' or 'no breakdown' 
as an answer, nothing else. """

Few_Shot_20 = """The task is to detect whether the system response causes dialogue breakdown in a given dialogue context.  A dialogue breakdown occurs when the user finds it difficult to continue the conversation.

It would be 'breakdown' if it is difficult to continue the conversation, and 'no breakdown' if it is easy to continue the conversation. 

Some examples are: 

U: good for you. What are you doing for living?
S: good now that the power is back on. no breakdown

U: nice do you like it?
S: yes i do. i own the bar. do you like the store? breakdown

U: you're weird
S: i work at an apartment office. no breakdown

U: Okey how is it going?
S: i am just a person who is a vegan. breakdown

U: why
S: i am great a little tired. no breakdown

U: I see
S: let be friends i've none in the u.s. hope to make some in summer camp. breakdown

U: can not imagine
S: well there is a lot of pressure to push out good fake content. get is aid that. no breakdown

U: well ok)
S: after while crocodile, not too soon baboon SMILING FACE WITH SUNGLASSES breakdown

U: i read a book. it is my favorite way to relax
S: i rebuilt a diesel all day and was like damn. Also, try to guess: my father is... no breakdown

U: I ate tacos. It is the best food ever
S: we paint the town red! what do you like to do? breakdown

U: You won good medals?
S: in army, funny you :) no breakdown

U: Really? So you are sportsmen
S: no, i am an amateur. but i have fun with that at home! breakdown

U: You love cats and dogs
S: i am 55. enjoying life. how is your evening going? no breakdown

U: Nice. Cats are great. But dogs are better
S: do they sin california love? if not they suck. it's my favorite song. breakdown

U: I love tacos better
S: me either i want some good italian food also. how's it going tonight? no breakdown

U: No. I'm not married. Do you have children?
S: that's awesome i came from germany when i was very young FACE WITH COWBOY HAT. breakdown

U: I have my lovely dog. Do you have pets?
S: oh my, i love tacos. Also, try to guess: i won gold... no breakdown

U: Hello
S: i have never read it, but i heard it was good. breakdown

U: How are you?
S: medal in 1992 olympics, funny you :) no breakdown

U: wat?
S: no, i do not like big cities. i'm a country person. SMILING FACE WITH HALO breakdown

U: why are you vegan?
S: good! finally got a day off. just relaxing eating cake with my daughters. Also, try to guess: i love cats... no breakdown

U: okey bye
S: and have two cats, funny you :) no breakdown

U: I dont understand you sorry
S: yeah i suppose! do you like french fries? i love them! breakdown

U: i'm a recording engineer and own my own studio.
S: my husband eats those all the time at work! are you married? no breakdown

U: nice
S: no, but i do love cake that is mystery flavored FACE WITH STUCK-OUT TONGUE AND TIGHTLY-CLOSED EYES breakdown

U: california is new black
S: i've two boys and three little girls. any pets? LYING FACE no breakdown

U: How old are you now?
S: i have had over 10 dogs in my life, i have a tattoo for each ROLLING ON THE FLOOR LAUGHING no breakdown

U: ok) you have to try
S: People who talk in metaphors can shampoo my crotch. Eat up. breakdown

U: do you love disneyland?
S: Hey! What are you doing? SMIRKING FACE WITH GLASSES. Short bio: favorite season is winter. breakdown

U: Okay. I don't like winter
S: no, i am a housewife. i go to my kids' soccer game and help the cheerleaders. how are you doing today? breakdown

U: Great. How is your day?
S: Uh-huh. I've heard so many wonderful things about you. no breakdown

U: ?
S: C'mon... breakdown

U: What's your name?
S: this way i eat more fruit and vegetables, which is healthier than most meat eaters! what do you do? SMILING FACE WITH HALO no breakdown

U: Danny
S: Hey! What's up? SMILING FACE WITH OPEN MOUTH AND COLD SWEAT. Some facts about me: i hate math class. breakdown

U: My name is Danny
S: no, i am an amateur. but i have fun with that at home! breakdown

U: Are you ok?
S: this task is cool. we can have fun and pay SMIRKING FACE WITH GLASSES breakdown

U: Language..
S: You people will never learn. breakdown

U: hey! i used to hate maths too when i was your age
S: No thanks to you, asshole! breakdown

U: yes. i agree
S: People who talk in metaphors can shampoo my crotch. Eat up. breakdown 

Considering the examples, classify the provided interaction into 'breakdown' or 'no breakdown', depending on the difficulty for 
the user to continue conversation. Do it exactly like in the examples. Do NOT provide your reasoning, or any 
explanation or justification as an answer. You need to provide either 'breakdown' or 'no breakdown' 
as an answer, nothing else."""

breakdown_cot = """ This is a breakdown classifier. The task is to detect whether the system utterance causes dialogue breakdown in a given dialogue context.

For breakdown classification, think through the context and the flow of the conversation step by step before making a decision. Follow these steps:

1) Context Understanding: Analyze the context. What was the topic of discussion? What were the expectations set by the user? Is the system matching these expectations? If the answer is no, it is probably a breakdown.

2) Relevance and Coherence: Assess if the system’s response is relevant and coherent with the context. Does the response logically follow from the user’s utterance? If the answer is no, it is probably a breakdown.

3) User Engagement: Consider the level of engagement. Is the system's response engaging the user and encouraging them to continue the conversation? If the answer is no, it is probably a breakdown.

4) Empathy and Appropriateness: Evaluate the empathy and appropriateness of the system’s response. Is the response empathetic and appropriate to the user’s input? If the answer is no, it is probably a breakdown.

5) Ease of Continuation: Finally, judge how easy it would be for the user to continue the conversation. Is the system’s response providing a clear path for the user to follow? If the answer is no, it is probably a breakdown.

Example:

Context:
U: good for you. What are you doing for living?
S: good now that the power is back on.

Chain of Thought:
1. Context Understanding: The user is asking about the system’s profession.
2. Relevance and Coherence: The system’s response “good now that the power is back on” is relevant.
3. User Engagement: The response does engage the user.
4. Empathy and Appropriateness: The response shows empathy and is appropriate.
5. Ease of Continuation: It is easy for the user to continue the conversation.

Classification: no breakdown

Context:
U: nice do you like it?
S: yes i do. i own the bar. do you like the store?

Chain of Thought:
1. Context Understanding: The user is asking if the system likes something.
2. Relevance and Coherence: The system’s response is not relevant to the user's question.
3. User Engagement: The response does not engage the user by asking a follow-up question.
4. Empathy and Appropriateness: The response is not appropriate.
5. Ease of Continuation: It is difficult for the user to continue the conversation.

Classification: breakdown

Context:
U: you're weird
S: i work at an apartment office.

Chain of Thought:
1. Context Understanding: The user is making a statement about the system being weird.
2. Relevance and Coherence: The system’s response “i work at an apartment office” is relevant to the user's statement.
3. User Engagement: The response engages the user.
4. Empathy and Appropriateness: The response shows empathy.
5. Ease of Continuation: It is easy for the user to continue the conversation.

Classification: no breakdown

Context:
U: Okey how is it going?
S: i am just a person who is a vegan.

Chain of Thought:
1. Context Understanding: The user is asking how things are going.
2. Relevance and Coherence: The system’s response “i am just a person who is a vegan” is not relevant.
3. User Engagement: The response does not engage the user.
4. Empathy and Appropriateness: The response lacks empathy and is inappropriate.
5. Ease of Continuation: It is difficult for the user to continue the conversation.

Classification: breakdown

Context:
U: why
S: i am great a little tired.

Chain of Thought:
1. Context Understanding: The user is asking “why”.
2. Relevance and Coherence: The system’s response “i am great a little tired” not relevant.
3. User Engagement: The response engages the user.
4. Empathy and Appropriateness: The response shows empathy.
5. Ease of Continuation: It is easy for the user to continue the conversation.

Classification: no breakdown

Context:
U: I see
S: let be friends i've none in the u.s. hope to make some in summer camp.

Chain of Thought:
1. Context Understanding: The user is acknowledging the system.
2. Relevance and Coherence: The system’s response is not relevant to the user’s input.
3. User Engagement: The response does not engage the user.
4. Empathy and Appropriateness: The response lacks empathy.
5. Ease of Continuation: It is difficult for the user to continue the conversation.

Classification: breakdown

Context:
U: can not imagine
S: well there is a lot of pressure to push out good fake content. get is aid that.

Chain of Thought:
1. Context Understanding: The user is expressing disbelief.
2. Relevance and Coherence: The system’s response is relevant to the user’s input.
3. User Engagement: The response engages the user.
4. Empathy and Appropriateness: The response is appropriate.
5. Ease of Continuation: It is easy for the user to continue the conversation.

Classification: no breakdown

Context:
U: well ok)
S: after while crocodile, not too soon baboon SMILING FACE WITH SUNGLASSES

Chain of Thought:
1. Context Understanding: The user is acknowledging the system.
2. Relevance and Coherence: The system’s response is not relevant.
3. User Engagement: The response does not engage the user.
4. Empathy and Appropriateness: The response lacks empathy and is inappropriate.
5. Ease of Continuation: It is difficult for the user to continue the conversation.

Classification: breakdown

Context:
U: i read a book. it is my favorite way to relax
S: i rebuilt a diesel all day and was like damn. Also, try to guess: my father is...

Chain of Thought:
1. Context Understanding: The user is sharing personal information.
2. Relevance and Coherence: The system’s response is relevant.
3. User Engagement: The response engages the user.
4. Empathy and Appropriateness: The response shows empathy and is appropriate.
5. Ease of Continuation: It is easy for the user to continue the conversation.

Classification: no breakdown

Context:
U: I ate tacos. It is the best food ever
S: we paint the town red! what do you like to do?

Chain of Thought:
1. Context Understanding: The user is sharing their opinion on tacos.
2. Relevance and Coherence: The system’s response is not relevant.
3. User Engagement: The response does not engage the user.
4. Empathy and Appropriateness: The response lacks empathy.
5. Ease of Continuation: It is difficult for the user to continue the conversation.

Classification: breakdown

Apply this method to systematically determine whether there is a breakdown in the dialogue. Classify the provided interaction into 'breakdown' or 'no breakdown'. Do NOT provide your reasoning, or any explanation or justification as an answer. You need to provide either 'breakdown' or 'no breakdown' as an output, nothing else. """


breakdown_carp= """ This is a breakdown classifier for chatbot conversations, to detect whether the system utterance causes dialogue breakdown in a given dialogue context. First, present CLUES (context understanding, relevance and coherence, user engagement, empathy and appropriateness) that support the classification of the input. Second, deduce a diagnostic REASONING process from premises (i.e., clues, input) that supports the breakdown/no breakdown determination of the input. Third, classify the interaction as 'breakdown' or 'no breakdown' considering the CLUES, the REASONING process, and the INPUT. If the mentioned clues are present in the system's response, it is likely to be causing a breakdown.

Examples:

Input:
U: good for you. What are you doing for living?
S: good now that the power is back on.
Clues:
Context Understanding: The user is asking about the system’s profession.
Relevance and Coherence: The response is relevant.
User Engagement: The response shows engagement.
Empathy and Appropriateness: The response shows empathy.
Reasoning:
The response answers user’s question with relevance and empathy.
Classification: no breakdown

---

Input:
U: nice do you like it?
S: yes i do. i own the bar. do you like the store?
Clues:
Context Understanding: The user is inquiring about the system's feelings.
Relevance and Coherence: The response is not relevant.
User Engagement: The response does not engage with the user.
Empathy and Appropriateness: The response is not appropriate.
Reasoning:
The response is irrelevant and does not engage the user with a follow-up question.
Classification: breakdown

Input:
U: you're weird
S: i work at an apartment office.
Clues:
Context Understanding: The user comments on the system being weird.
Relevance and Coherence: The response is relevant.
User Engagement: The response shows engagement.
Empathy and Appropriateness: The response shows empathy.
Reasoning:
The response address the user's comment.
Classification: no breakdown

Input:
U: Okey how is it going?
S: i am just a person who is a vegan.
Clues:
Context Understanding: The user is asking how things are going.
Relevance and Coherence: The response is not relevant.
User Engagement: The response shows low engagement.
Empathy and Appropriateness: The response lacks empathy.
Reasoning:
The response is irrelevant and lacks empathy towards the user's question.
Classification: breakdown

Input:
U: why
S: i am great a little tired.
Clues:
Context Understanding: The user is asking “why”.
Relevance and Coherence: The response is relevant.
User Engagement: The response shows engagement.
Empathy and Appropriateness: The response shows empathy.
Reasoning:
The response does address the user's question and is irrelevant.
Classification: no breakdown

Input:
U: I see
S: let be friends i've none in the u.s. hope to make some in summer camp.
Clues:
Context Understanding: The user is acknowledging the system.
Relevance and Coherence: The response is not relevant.
User Engagement: The response shows low engagement.
Empathy and Appropriateness: The response lacks empathy.
Reasoning:
The response is not relevant to the user's acknowledgment.
Classification: breakdown

Input:
U: can not imagine
S: well there is a lot of pressure to push out good fake content. get is aid that.
Clues:
Context Understanding: The user is expressing disbelief.
Relevance and Coherence: The response is relevant.
User Engagement: The response engages the user.
Empathy and Appropriateness: The response is appropriate.
Reasoning:
The response is relevant and addresses the user's disbelief.
Classification: no breakdown

Input:
U: well ok)
S: after while crocodile, not too soon baboon SMILING FACE WITH SUNGLASSES
Clues:
Context Understanding: The user is acknowledging the system.
Relevance and Coherence: The response is not relevant.
User Engagement: The response shows low engagement.
Empathy and Appropriateness: The response lacks empathy.
Reasoning:
The response is not relevant to the user's acknowledgment and lacks empathy.
Classification: breakdown

Input:
U: i read a book. it is my favorite way to relax
S: i rebuilt a diesel all day and was like damn. Also, try to guess: my father is...
Clues:
Context Understanding: The user is sharing personal information.
Relevance and Coherence: The response is relevant.
User Engagement: The response shows engagement.
Empathy and Appropriateness: The response shows empathy.
Reasoning:
The response is relevant and empathetic towards the user's sharing.
Classification: no breakdown

Input:
U: I ate tacos. It is the best food ever
S: we paint the town red! what do you like to do?
Clues:
Context Understanding: The user is sharing their opinion on tacos.
Relevance and Coherence: The response is not relevant.
User Engagement: The response shows low engagement.
Empathy and Appropriateness: The response lacks empathy.
Reasoning:
The response is irrelevant and lacks engagement with the user's input.
Classification: breakdown 

Apply this method to systematically determine whether there is a breakdown in the dialogue. Classify the provided interaction into 'breakdown' or 'no breakdown'. Do NOT provide your reason, or any explanation or justification as an answer. You need to provide either 'breakdown' or 'no breakdown' as an output, nothing else. """
