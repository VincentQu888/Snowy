#file reading imports
from dotenv import load_dotenv
import configparser
import pickle 
import json

#discord imports
import discord
from discord import Intents, Client, Message, app_commands, Interaction, TextChannel, Role, utils
from discord.ext import commands

#instagram imports
import instaloader

#model imports
from transformer_model import stem, transformer, feed_forward, gen_pe, attention, multi_head_attn
import concurrent.futures
import spacy
import en_core_web_md
import torch
import numpy as np

#misc
from typing import Final
import time
import os
import asyncio


#load discord token
load_dotenv()
TOKEN: Final[str] = os.getenv('DISCORD_TOKEN')


#discord bot settings
intents: Intents = Intents.default()
intents.message_content = True 
discord_bot = commands.Bot(command_prefix="!", intents=intents)


#storing channels for each server
announcement_channels = {}



#discord bot events and commands

@discord_bot.event
async def on_ready() -> None:
    '''
    '''
    print(f"{discord_bot.user} is now running")

    try:
        synced = await discord_bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")

        #await check_posts()
    except Exception as e:
        print(e) 



@discord_bot.event
async def on_guild_join(guild):
    '''
    '''
    for channel in guild.channels:
        if isinstance(channel, TextChannel):
            await channel.send("Hey there, I'm Snowy! A snow day predictor for YRDSB. Set me up by using the /setinfo command to set up the channel I will announce snow days in, and the role it will ping. For more info, read: _\_\_\_\_")
            break




@discord_bot.tree.command(name="setinfo", description="Set info for Snowy to announce snow days in!")
@app_commands.describe(channel = "Channel Name", role = "Role Name")
async def setinfo(interaction: Interaction, channel: TextChannel, role: Role):
    '''
    '''
    if interaction.user.guild_permissions.administrator:

        global announcement_channels
        announcement_channels[interaction.guild.id] = (channel.id, role.id)

        with open("announcement_channels.pkl", "wb") as file:
            pickle.dump(announcement_channels, file)

        await interaction.response.send_message(f"Set snow day announcement channel to: {channel.mention}, and role ping to {role.name}.")
        
    else:
        await interaction.response.send_message(f"Sorry {interaction.user.mention}, you don't have permissions to use this command!")



@discord_bot.tree.command(name="test", description="Sends a message in channel set through /setinfo to test if info is set correctly.")
async def test(interaction: Interaction):
    '''
    '''
    if interaction.user.guild_permissions.administrator:
        
        with open("announcement_channels.pkl", "rb") as file:
            global announcement_channels
            announcement_channels = pickle.load(file)


        if interaction.guild.id in announcement_channels.keys():
            guild = interaction.guild
            channel = utils.get(guild.channels, id=announcement_channels[guild.id][0])
            role = utils.get(guild.roles, id=announcement_channels[guild.id][1])
            
            await channel.send(f"{role.mention} I'm sending messages and snow day announcements in this channel!")
        else:
            await interaction.response.send_message(f"{interaction.user.mention} info not set :(")

    else:
        await interaction.response.send_message(f"Sorry {interaction.user.mention}, you don't have permissions to use this command!")




async def announce(msg):
    '''
    '''
    with open("announcement_channels.pkl", "rb") as file:
        global announcement_channels
        announcement_channels = pickle.load(file)

    for guild in discord_bot.guilds:
        if guild.id in announcement_channels.keys():
            channel = utils.get(guild.channels, id=announcement_channels[guild.id][0])
            role = utils.get(guild.roles, id=announcement_channels[guild.id][1])

            await channel.send(f"{role.mention} {msg}")



def run_bot():
    '''
    '''
    discord_bot.run(token=TOKEN)





def new_post(bot, last_id):
    '''
    '''
    #get posts from yrdsb
    profile = instaloader.Profile.from_username(bot.context, 'yrdsb.schools')
    posts = profile.get_posts()

    for i, post in enumerate(posts):
        if i > 0:
            break

        if post.mediaid != last_id:
            return post
        
    return False




def check_posts():
    '''
    '''
    #parse config info
    config = configparser.ConfigParser()
    config.read('F:\ics4u\projects\capstone\config_info.ini')



    while True:
        insta_bot = instaloader.Instaloader()
        insta_bot.login(config['instagram']['username'], config['instagram']['password'])

        last_id = ""
        post = new_post(insta_bot, last_id)


        if post:
            #plug new post into model to check for snow day
            test_x = [
                post.caption.replace(",", "")
            ]

            map(stem, test_x) #stem
            test_doc = nlp(test_x) #vectorize 

            test_x = [token.vector for token in test_doc]
                    

            #prediction
            pred = model.predict(torch.tensor(np.array(input)), False) #prediction
            if True: #some condition, depends on model performance
                announce(f"{pred[0]}")



        time.sleep(60*10)





#driver code
nlp = spacy.load("en_core_web_md")

#opening file with model
with open("transformer.pkl", "rb") as file:
    model = pickle.load(file)


run_bot()