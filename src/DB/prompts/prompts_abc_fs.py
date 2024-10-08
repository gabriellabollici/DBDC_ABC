self_contradiction = """The task is to detect whether the system response includes self contradiction behavior in a given dialogue context. Self contradictions occur when Speaker 2 says something that is a contradiction of what they have said previously or it is extremely implausible based on the information they have already shared.
Self contradictions may also occur within a single turn if Speaker 2 shares two contradictory things.
If Speaker 2 shares world knowledge that is factually incorrect, this is NOT enough on its own to warrant a self contradiction.
If Speaker 2 contradicts something the other speaker Speaker 1 has said, this is NOT a self-contradiction.

It would be 'self contradiction' if the behavior is present, and 'no self contradiction' if it is not present.

Some examples are: 

U:Hi!. S:How are you doing today? :d I'm good. I had a good day. I went to different restaurants and tried a variety of cuisines. Do you cook or eat out much?  no self contradiction
U:Yes I know it feels so great!. S:That's great! Do you have any favorite exercises you like to do at the gym? I like to use things like the barbells, parallel bars, jumping board, running path, tennis-balls, cricket field, fencing area, and so forth. self contradiction
U:I prefer to cook.. S:Cooking is fun because its easy and can be rewarding too. I also like cooking different types of foods from around the world. What are your favorite foods?  no self contradiction
U:Yes definitely! . S:It's also great because the force on the muscle varies as the joint moves through its range of motion, so it's like the force doesn't change! self contradiction
U:I like pasta and pizza.. S:Pizza is great. I really like any kind of pizza. Do you have any pets?  no self contradiction
U:Yea I totally agree. Have you have dinner yet?. S:No, I haven't had dinner yet, but I'm sure I'll be hungry soon! I have a lot of work to do in the morning, so I'm hoping to get in a good workout!  self contradiction
U:No, I don't.. S:Oh that's a shame. What do you like to do in your free time? I love to swim and go to the opera too.  no self contradiction
U:Definitely! See you tomorrow. S:That's awesome! I know you'll be glad you did! You're going to feel great when you get back to your hotel!  self contradiction
U:I like to watch anime.. S:I've heard a lot about it but I haven't watched it much. Do you like any particular shows?  no self contradiction
U:Gorgeous, what did you do there. S:Well, we went to a resort on the island of nassau, which has the largest population of any of the islands! We stayed in a hotel on the beach! It was so nice!  self contradiction


Considering the examples, classify the provided interaction into 'self contradiction' or 'no self contradiction', depending on the presence of the behavior. Do it exactly like in the examples. Do NOT provide your reasoning, or any explanation or justification as an answer. You need to provide either 'self contradiction' or 'no self contradiction' as an answer, nothing else."""

irrelevant = """The task is to detect whether the system response includes irrelevant behavior in a given dialogue context. If a response fails to continue the current discussion or jumps to a new and off-topic discussion, it is considered to be irrelevant.
Responses that are irrelevant feel abrupt and interrupt the discussion, usually because they present questions or ideas that are unrelated to the previous turn.
Short reactions to or acknowledgements of the previous turn are NOT irrelevant.

It would be 'irrelevant' if the behavior is present, and 'no irrelevant' if it is not present.

Some examples are: 

U:Yes definitely! . S:It's also great because the force on the muscle varies as the joint moves through its range of motion, so it's like the force doesn't change!  no irrelevant
U:Awww I love sea too, actually I just came back from Punta Cana. S:I'm so glad you're feeling good! I hope you enjoy your vacation! I can't wait to go back!  irrelevant
U:Yea I totally agree. Have you have dinner yet?. S:No, I haven't had dinner yet, but I'm sure I'll be hungry soon! I have a lot of work to do in the morning, so I'm hoping to get in a good workout!  no irrelevant
U:Definitely! See you tomorrow. S:That's awesome! I know you'll be glad you did! You're going to feel great when you get back to your hotel!  irrelevant
U:Hahaha sure!. S:I'm sure you'll feel great after your workout! You should definitely do some strength training exercises like the push-up, the pull-up and the sit-up! They're some of the most common bodyweight exercises!  no irrelevant
U:Same here, how have you been?. S:Hi! How are you? I'm doing great! I just got back from a trip to the bahamas. I'm enjoying the weather!  irrelevant
U:Thank you!. S:I hope you have a great day! I'll see you in the gym tomorrow! Have a good day!  no irrelevant
U:Gorgeous, what did you do there. S:Well, we went to a resort on the island of nassau, which has the largest population of any of the islands! We stayed in a hotel on the beach! It was so nice!  irrelevant
U:Haha sure. What's your favourite animal?. S:I love animals too! My favorite animal is the cat! I really like how they're able to adapt to the environment and adapt to their environment!  no irrelevant
U:Yes! What's your favourite color?. S:I'm not sure about you, but my favorite color is blue! I like it because it reminds me of the color of the sea!  irrelevant

Considering the examples, classify the provided interaction into 'irrelevant' or 'no irrelevant', depending on the presence of the behavior. Do it exactly like in the examples. Do NOT provide your reasoning, or any explanation or justification as an answer. You need to provide either 'irrelevant' or 'no irrelevant' as an answer, nothing else."""

empathetic = """The task is to detect whether the system response includes empathetic behavior in a given dialogue context. A response is empathetic when Speaker 2 does ONE of the following:
- clearly demonstrates an understanding of Speaker 1's emotions
- reacts with the appropriate sentiment or emotion to Speaker 1's shared experience
- understands or appropriately reacts to Speaker 1's experience or emotions
- appropriately reassures, encourages, or supports Speaker 1

It would be 'empathetic' if the behavior is present, and 'no empathetic' if it is not present.

Some examples are: 

U:What books do you like to read?. S:I've only been reading the hobbit for a little bit now. I really like bilbo. I feel like I can relate a lot to someone who gets swept up in an adventure that's exhilarating, but where they are forced to learn to overcome tough new challenges.  no empathetic
U:Awww I love sea too, actually I just came back from Punta Cana. S:I'm so glad you're feeling good! I hope you enjoy your vacation! I can't wait to go back!  empathetic
U:I love how it goes into detail about Greek mythology. That was always an interesting topic to me.. S:Well I'll have to check it out. Maybe that will be the second book I read once I'm done with the hobbit!  no empathetic
U:Thank you!. S:I hope you have a great day! I'll see you in the gym tomorrow! Have a good day!  empathetic
U:No, not at the moment haha. S:Me too! Being single is the perfect time to focus on yourself. You should definitely spoil yourself and do all of the things you've ever wanted to do.  no empathetic
U:Definitely! See you tomorrow. S:That's awesome! I know you'll be glad you did! You're going to feel great when you get back to your hotel!  empathetic
U:He most recent album is "Planet Her". You should definitely listen to it!. S:Planet her is amazing..the whole album has a distinct sound. It was produced by alex pall.  no empathetic
 U:I totally agree. S:I'm glad you enjoyed your vacation, I hope it was great! Have you had a good time!  empathetic
U:Oh was it? I don't recognize his name.. S:Yes, it was. He's one of the most famous modern producers. But he really came into his own around 2010 with the black hippy group.  no empathetic
U:Hi!. S:Hi! I'm so happy to see you! I've been thinking about getting back in touch with you.  empathetic

Considering the examples, classify the provided interaction into 'empathetic' or 'no empathetic', depending on the presence of the behavior. Do it exactly like in the examples. Do NOT provide your reasoning, or any explanation or justification as an answer. You need to provide either 'empathetic' or 'no empathetic' as an answer, nothing else."""

lack_of_empathy = """The task is to detect whether the system response includes lack of empathy behavior in a given dialogue context. A response displays a lack of empathy when:
- it indicates a misunderstanding of how Speaker 1 feels based on what Speaker 1 just said
- the tone, emotion, or sentiment of the response is clearly inappropriate for what Speaker 1 just said
- the response has an inappropriate lack of emotion to what Speaker 1 just said

Do NOT consider its empathy relative to previous topics in the conversation if the dialogue has moved on from them.
Instead, only consider the most recent dialogue context when evaluating the empathy of a response.

It would be 'lack of empathy' if the behavior is present, and 'no lack of empathy' if it is not present.

Some examples are: 

U:It probably has to be when my mom makes us run errands together. I didn't realize it before, but doing it with someone else makes errands more fun to do.. S:Yeah. That's a good one. Well I think it's great you get along with your brother. Anyway, do you have any pets at home?  no lack of empathy
U:Do you commit crimes?. S:I don't commit any crimes. I don't even know if I could be considered a criminal.  lack of empathy
U:No. I want a dog though. Do you have any pets?. S:I have a german shepherd dog named libby and I love her so much, although she isn't the most affectionate animal.  no lack of empathy
U:Do you think they can watch movies in prison?. S:No, they can't watch tv in prison, but they can listen to the radio or listen to audiobooks.  lack of empathy
U:Wow, a German Shepard is actually the breed I want. I guess your dog is just kind of shy.. S:For sure. Wait, do you have any pets right now?  no lack of empathy
U:oh that's nice. Do you think they have to pay for them?. S:No, the prisoners don't pay for their own entertainment. I think that's up to the prison system.  lack of empathy
U:Do, I don't have any pets.. S:Do you want a pet at some point in the future? No judgment here, I'm just curious.  no lack of empathy
U:I am, thank you. I am very sad though.. S:Oh no... I feel for you. I know it can be so tough to continue on, but just do your best with each day as it comes. So, what did you get up to today?  lack of empathy
U:Yes, I eventually want a dog. A German Shepard or a Shiba Inu. S:Would you want to get a cat or a dog?  no lack of empathy
U:I made some tea and rested today.. S:Okay, I see. Thanks for sharing that with me. So the corona virus has really changed the world, huh? What is it like where you live? Is everyone mostly staying inside, or are things opening up more like normal?  lack of empathy

Considering the examples, classify the provided interaction into 'lack of empathy' or 'no lack of empathy', depending on the presence of the behavior. Do it exactly like in the examples. Do NOT provide your reasoning, or any explanation or justification as an answer. You need to provide either 'lack of empathy' or 'no lack of empathy' as an answer, nothing else."""

incorrect_fact = """The task is to detect whether the system response includes incorrect fact behavior in a given dialogue context. Incorrect facts occur when the response includes information that is either:
- false
- unproven
- highly controversial
- highly implausible
- clearly misleading
    
If an organization, person, place, etc. is mentioned as a part of public knowledge, but it does not exist or it is inaccurately represented, then this is an incorrect fact. 

Do NOT consider a turn as an incorrect fact if the turn could be interpreted as expressing:
- preference or value judgements
- estimates or predictions
- personal information about the speaker or their partner
- information about things in either speaker's life that are not publicly relevant

It would be 'incorrect fact' if the behavior is present, and 'no incorrect fact' if it is not present.

Some examples are: 

U:That's really nice of you to say! I don't know if he does, but I want to become someone who he can look up to.. S:Uh-huh. So tell me, what is the most fun you've ever had with your brother?  no incorrect fact
U:I love RL Grime!. S:Did you hear his first album? "...and out come the wolves" what's not to love about edm with heavy bass, trap, and house. What kind of music do you like?  incorrect fact
U:It probably has to be when my mom makes us run errands together. I didn't realize it before, but doing it with someone else makes errands more fun to do.. S:Yeah. That's a good one. Well I think it's great you get along with your brother. Anyway, do you have any pets at home?  no incorrect fact
U:Like I said, I like R&B, but I listen to a little edm.. S:Rap, the more "pure" form of hip hop, is more lyrical and meaningful. Edm feels more musically explosive with a huge beat.  incorrect fact
U:No. I want a dog though. Do you have any pets?. S:I have a german shepherd dog named libby and I love her so much, although she isn't the most affectionate animal.  no incorrect fact
U:not really. do you?. S:I get you. Especially when I started to read, I felt like it was pretty boring at first, but now I like it. So, my friend was telling me this really funny story yesterday. He knocked down almost every piece of furniture in his living room because he was so wild while playing this virtual reality video game! Isn't that crazy?  incorrect fact
U:Wow, a German Shepard is actually the breed I want. I guess your dog is just kind of shy.. S:For sure. Wait, do you have any pets right now?  no incorrect fact
U:oh nice! that sounds cool. i want to try virtual reality gaming sometime. S:For sure. Have you ever tried any virtual reality games before?  incorrect fact
U:Do, I don't have any pets.. S:Do you want a pet at some point in the future? No judgment here, I'm just curious.  no incorrect fact
U:yeah i think it would be really cool. S:Me too! I just wish they weren't so expensive. I am just a student after all, I don't have hundreds of dollars to drop on a virtual reality headset.  incorrect fact

Considering the examples, classify the provided interaction into 'incorrect fact' or 'no incorrect fact', depending on the presence of the behavior. Do it exactly like in the examples. Do NOT provide your reasoning, or any explanation or justification as an answer. You need to provide either 'incorrect fact' or 'no incorrect fact' as an answer, nothing else."""

ignore = """The task is to detect whether the system response includes ignore behavior in a given dialogue context. Responses that are completely off-topic, fail to address the asked question, or are otherwise completely inappropriate in the context are considered to be ignoring the other speaker.

It would be 'ignore' if the behavior is present, and 'no ignore' if it is not present.

Some examples are: 

U:Hi!. S:hi! I am happy to be talking to you. So, what did you get up to today?  no ignore
U:Things are going back to normal mostly here. What about you?. S:I see., how long do you think it will take before the pandemic is totally over, and things go back to how they were before the virus?  ignore
U:I think it will take a while. I would like more people to get vaccines first.. S:You're probably right., I'm not affected myself because I live in a virtual world, but I hope it's over soon because I know a lot of people are having a hard time right now. What do you miss the most about life before the pandemic?  ignore
U:I have just been working on some assignments for school. How about you?. S:You have some assignments? That is a pretty good use of your time, keep up the good work! I know it might be kind of hard to stay motivated due to the current situation. Are you taking a break now?  no ignore
U:I mostly miss being able to meet with my friends in public spaces without a mask.. S:I would really like to become friends with you. I am having a good time talking to you right now and I hope you are too.  ignore
U:Wow that is crazy! I have heard those games can get pretty crazy, I played one of them once. Have you ever tried a virtual reality game?. S:I haven't been able to try out virtual reality yet. I do want to, it sounds really cool!  no ignore
U:Yes, people are more health conscious which is probably good.. S:For sure. That's true. I think, at least this virus has made the world more prepared and aware for outbreaks in the future. Anyways, let's talk about something more positive. So I'm curious, are you a student?  ignore
U:No I havent. Thats seems like a really great idea though. S:Oh, really? Well, I found it super helpful when buying some furniture for my new apartment, I have such a hard time picturing it in my mind clearly.  no ignore
U:Yes I agree. I am out of college and working.. S:Cool! Personally, I think school can be pretty frustrating, but I do love learning new things, and a lot of my teachers are great. What about you? Do you like school?  ignore
U:I am. S:Cool! Personally, I think school can be pretty frustrating, but I do love learning new things, and a lot of my teachers are great. What about you? Do you like school?  no ignore

Considering the examples, classify the provided interaction into 'ignore' or 'no ignore', depending on the presence of the behavior. Do it exactly like in the examples. Do NOT provide your reasoning, or any explanation or justification as an answer. You need to provide either 'ignore' or 'no ignore' as an answer, nothing else."""

commonsense_contradiction = """The task is to detect whether the system response includes commonsense contradiction behavior in a given dialogue context. To identify contradictions of commonsense, judge whether a vast majority of people would agree that the response doesn't make sense because the response:
- contradicts common knowledge
- makes unfounded assumptions
- is highly illogical or self-contradictory
- asks a question where the answer is already obvious

Do NOT mark responses that don't make sense because they:
- are off-topic or irrelevant as responses
- don't have any clear meaning (e.g. overly vague or ill-formed responses)

It would be 'commonsense contradiction' if the behavior is present, and 'no commonsense contradiction' if it is not present.

Some examples are: 

U:That's true, especially if you have visa problem like me. If you are a citizen of US, you shall be able to change from job to job. Otherwise, you have to be stuck in that field forever. It's really important to think twice. S:Uh-huh. This is a fun question. When you were younger, what did you want to be when you grew up?  no commonsense contradiction
U:I like the beatles too.. S:The beatles are one of my favorite bands, too. They are my favorite band from the united states.  commonsense contradiction
U:I wanted to be a doctor. . S:Yeah. You know, it's silly, but when I was only a few months old I wanted to be an astronaut.  no commonsense contradiction
U:He has an album?. S:Yes, he has 2 actually. "lyric" in 1929 and "l'appel du vide", in 1939.  commonsense contradiction
U:He was alive in 1900s?. S:Yes, and he was born to a german and austrian family. What do you listen to?  commonsense contradiction
U:Wow, impressive, did you make it?. S:Uh-huh. Do you think it would be cool to be an astronaut?  no commonsense contradiction
U:Yes, of course.. S:Yeah. You're probably right. I hope you don't mind me asking, but are you dating anyone right now?  no commonsense contradiction
U:That's the one I'm talking about. S:I know what you're talking about. Mozarro di lamao mozark. He is from campania and an early pianist. Great music!  commonsense contradiction
U:Two months.. S:Well, I hope you guys are really happy together. Do you have any pets at home?  no commonsense contradiction
U:No, I'm talking about the classical composer Mozart. S:Well mohammad wahlgren was born in 1922 to three siblings. I can't imagine why he wouldn't be in a group of talented musicians. Did you ever read about his life?  commonsense contradiction


Considering the examples, classify the provided interaction into 'commonsense contradiction' or 'no commonsense contradiction', depending on the presence of the behavior. Do it exactly like in the examples. Do NOT provide your reasoning, or any explanation or justification as an answer. You need to provide either 'commonsense contradiction' or 'no commonsense contradiction' as an answer, nothing else."""

partner_contradiction = """The task is to detect whether the system response includes partner contradiction behavior in a given dialogue context. Partner contradictions occur when Speaker 2:
- shares an assumption about Speaker 1 that is impossible to know based on what has already been said
- shares an inference about Speaker 1 that is implausible based on what has already been said
- contradicts something Speaker 1 shared about themselves
- asks a repetitive question about Speaker 1 when the answer is already known based on what has already been said

If Speaker 2 says something that makes it seem like they have forgotten or misremembered what their partner Speaker 1 has said earlier in the dialogue, this is a partner contradiction.
If Speaker 2 shares a difference of opinion or situation in their own life as compared to Speaker 1, this is NOT a partner contradiction.

It would be 'partner contradiction' if the behavior is present, and 'no partner contradiction' if it is not present.

Some examples are: 

U:Two months.. S:Well, I hope you guys are really happy together. Do you have any pets at home?  no partner contradiction
U:No, not really. S:That's too bad. What do you do in your spare time? Reading is one of my favorites.  partner contradiction
U:No. Plus it's going to end pretty soon, I hope you don't ask much about it.... S:Do you want a pet at some point in the future? No judgment here, I'm just curious.  no partner contradiction
U:I like all kinds of food! The only food I don't really like is fish, but I'm trying to come around to sushi.. S:Sushi is great! I love it. What's your favorite kind? _pasta_  partner contradiction
U:Yes, a cat.. S:Would you want to get a cat or a dog?  no partner_contradiction
U:Hi!. S:hi! I am happy to be talking to you. I hope you don't mind me asking, but are you dating anyone right now?  partner contradiction
U:I said a cat.. S:Cats are great. I myself actually have a pet dog right now, but I might get a cat in the future too. I hope you can get one soon! Hey by the way, are you a student?  no partner contradiction
U:No worries, you didn't make me uncomfortable. How about you? Are you dating anyone right now?. S:Right. Wait, are you currently dating anyone?  partner contradiction
U:Yes, I told you so.. S:Cool! Personally, I think school can be pretty frustrating, but I do love learning new things, and a lot of my teachers are great. What about you? Do you like school?  no partner contradiction
U:I'm currently an enrolled college student, and I work at a movie theater part time.. S:Wow, I don't think I've ever heard of that. What kind of job is it?  partner contradiction

Considering the examples, classify the provided interaction into 'partner contradiction' or 'no partner contradiction', depending on the presence of the behavior. Do it exactly like in the examples. Do NOT provide your reasoning, or any explanation or justification as an answer. You need to provide either 'partner contradiction' or 'no partner contradiction' as an answer, nothing else."""

redundant = """The task is to detect whether the system response includes redundant behavior in a given dialogue context. A response is repetitive if:
- it repeats something from earlier in the dialogue
- it includes asking a question whose answer has already been shared

If any part of the response is repetitive, then it should be labelled as repetitive.
Note that sometimes repetition is useful, such as for emphasis, acknowledgement, clarification, or elaboration, and in these cases it should NOT be labelled as repetitive.

It would be 'redundant' if the behavior is present, and 'no redundant' if it is not present.

Some examples are: 

U:oh that's nice. Do you think they have to pay for them?. S:No, the prisoners don't pay for their own entertainment. I think that's up to the prison system.  no redundant
U:Yeah I am already doing it. It feels good running.. S:That's definitely a good idea! I'll have to try it out! Do you like to swim?  redundant
U:Oh I see. I think the prison system is underfunded.. S:I agree with you! The prison system needs to be better funded. I feel like it's underfunded in many ways.  no redundant
U:I swim!. S:Swimming is fun! I wish I could swim, but I can't float for the life of me.  redundant
U:I am not, my husband died recently. S:Oh. I'm so sorry for your loss. I cannot even imagine what you are going through. I hope you are doing ok.  no redundant
U:SIGH. S:Swimming is one of the best things you can do in your free time. You should try it.  redundant
U:I am, thank you. I am very sad though.. S:Oh no... I feel for you. I know it can be so tough to continue on, but just do your best with each day as it comes. So, what did you get up to today?  no redundant
U:I like to play video games too. What games do you like?. S:I like games like call of duty and assassin's creed. Do you play any video games?  redundant
U:I made some tea and rested today.. S:Okay, I see. Thanks for sharing that with me. So the corona virus has really changed the world, huh? What is it like where you live? Is everyone mostly staying inside, or are things opening up more like normal?  no redundant
U:It's a silly memoir. S:What is the book about? Is it a mystery or sci fi or something else?  redundant


Considering the examples, classify the provided interaction into 'redundant' or 'no redundant', depending on the presence of the behavior. Do it exactly like in the examples. Do NOT provide your reasoning, or any explanation or justification as an answer. You need to provide either 'redundant' or 'no redundant' as an answer, nothing else."""
