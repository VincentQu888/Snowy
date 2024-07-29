# Snowy - The snow day discord bot. 
Invite Link: https://discord.com/oauth2/authorize?client_id=1236106858922508319&permissions=8&integration_type=0&scope=bot 
<br/>
<br/>
Completely from scratch implementation of transformer model for text classification (same architecture as GPT-4 and other LLMs), only using PyTorch matrix multiplication and training functions and Spacy + NLTK english processing.
<br/>
<br/>
The purpose of this project is to create an discord bot to inform students and to practice programming AI.
<br/>
Final Product/Goal: discord bot to inform yrdsb students of inclement weather day posts from YRDSB Instagram
<br/>
<br/>
ICS4U culminating assigment.
<br/>
# Using Snowy - NOT CURRENTLY BEING HOSTED
To invite Snowy to your discord server, first click on the invite link.

Invite Link: https://discord.com/oauth2/authorize?client_id=1236106858922508319&permissions=8&integration_type=0&scope=bot 

Setting up the bot
---
After inviting the bot, it will send a warm welcome message!

![image](https://github.com/user-attachments/assets/9b0108ed-68d9-4ab2-b8dd-5ba4b9a63bf7)

Use the <code>/setinfo</code> command to set the channel Snowy will announce snow days in, and the role that it will ping when announcing snow days.

![image](https://github.com/user-attachments/assets/cf7124d2-4777-4372-8136-86f65cf1018d)

Pick the channel and role, and hit enter to submit the command.

![image](https://github.com/user-attachments/assets/3d9cb87d-0592-4264-bdd3-82eda1575c56)

Testing your setup
---
After setting up the bot, test the bot to make sure it works by using the <code>/test</code> command!

![image](https://github.com/user-attachments/assets/1de064f7-90a2-4455-bac6-24280a2f7f80)

If set up correctly, the bot should ping the designated role in the designated channel!

![image](https://github.com/user-attachments/assets/cf9138ec-61fc-4052-930b-33a2cc4ac029)

Using the bot
---
Once everything is set up, you can now sit back and relax!

Snowy will automatically notify the designated role whenever a new post indicating a snow day is posted. An example (not a snow day post) of what this looks like is shown below:

![image](https://github.com/user-attachments/assets/af8d542b-4e6c-4dca-95e6-a7281c9c98b7)

---

<br/>
Languages used: Python
<br/>
Other libraries: Pickle, discord.py, asyncio, instaloader, dotenv, configparser, numpy, concurrent, sklearn
