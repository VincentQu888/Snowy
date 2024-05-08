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
        On ready event of discord bot, acitvates and runs discord bot

        Args:
            None
            
        Returns:
            None
    '''
    
    print(f"{discord_bot.user} is now running")

    try:
        synced = await discord_bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")

        #await check_posts() #run bot process to wait for new posts
    except Exception as e:
        print(e) 



@discord_bot.event
async def on_guild_join(guild):
    '''
        On guild join event, activates on guild join and sends a introduction message
        
        Args:
            guild (discord guild): guild bot has joined
            
        Returns:
            None
    '''
    
    for channel in guild.channels:
        if isinstance(channel, TextChannel):
            await channel.send("Hey there, I'm Snowy! A snow day predictor for YRDSB. Set me up by using the /setinfo command to set up the channel I will announce snow days in, and the role it will ping. For more info, read: _\_\_\_\_")
            break




@discord_bot.tree.command(name="setinfo", description="Set info for Snowy to announce snow days in! (admin only)")
@app_commands.describe(channel = "Channel Name", role = "Role Name")
async def setinfo(interaction: Interaction, channel: TextChannel, role: Role):
    '''
        Set info command, sets the channel to announce snow days in and role to ping when announcing snow days
        
        Args:
            interaction (Interaction): command interaction from user
            channel (TextChannel): channel to set to
            role (Role): role to ping
            
        Returns:
            None
    '''
    
    if interaction.user.guild_permissions.administrator: #only admins allowed to use command

        global announcement_channels
        announcement_channels[interaction.guild.id] = (channel.id, role.id) #update announcement info

        with open("announcement_channels.pkl", "wb") as file: #save info to file for permanent storage
            pickle.dump(announcement_channels, file)

        await interaction.response.send_message(f"Set snow day announcement channel to: {channel.mention}, and role ping to {role.name}.") #return response
        
    else:
        await interaction.response.send_message(f"Sorry {interaction.user.mention}, you don't have permissions to use this command!") #no admin response



@discord_bot.tree.command(name="test", description="Sends a message in channel set through /setinfo to test if info is set correctly. (admin only)")
async def test(interaction: Interaction):
    '''
        Testing command, will send a message in set channel to test if server info was set correctly
        
        Args:
            interaction (Interaction): command interaction from user
            
        Returns:
            None
    '''
    
    if interaction.user.guild_permissions.administrator: #admin only command

        #update announcement info to most recently saved in file
        with open("announcement_channels.pkl", "rb") as file: 
            global announcement_channels
            announcement_channels = pickle.load(file)

        
        #check if guild info has been set
        if interaction.guild.id in announcement_channels.keys(): 
            
            #obtain guild, channel, and role objects to use in response
            guild = interaction.guild 
            channel = utils.get(guild.channels, id=announcement_channels[guild.id][0])
            role = utils.get(guild.roles, id=announcement_channels[guild.id][1])
            
            await channel.send(f"{role.mention} I'm sending messages and snow day announcements in this channel!") #successful response
        else:
            await interaction.response.send_message(f"{interaction.user.mention} info not set :(") #info not set response

    else:
        await interaction.response.send_message(f"Sorry {interaction.user.mention}, you don't have permissions to use this command!") #no admin response




async def announce(msg):
    '''
        Announcement function to announce when snow days occur.
        
        Args:
            msg (string): message to announce in all servers
            
        Returns:
            None
    '''
    
    with open("announcement_channels.pkl", "rb") as file: #update announcement info
        global announcement_channels
        announcement_channels = pickle.load(file)

    for guild in discord_bot.guilds: 
        if guild.id in announcement_channels.keys():
            #get channel and role objects
            channel = utils.get(guild.channels, id=announcement_channels[guild.id][0])
            role = utils.get(guild.roles, id=announcement_channels[guild.id][1])

            await channel.send(f"{role.mention} {msg}") #send message in channel pinging role




def run_bot():
    '''
        Function to run the bot
        
        Args:
            None
            
        Returns:
            None
    '''
    
    discord_bot.run(token=TOKEN)





def new_post(bot, last_id):
    '''
        Checks if there's a new post from YRDSB instagram
        
        Args:
            bot (instaloader): instagram bot logged into an account
            last_id (str): id of most recent post downloaded
            
        Returns:
            Post (instagram post): most recent post if post id was different, False (boolean): if post id was the same 
    '''
    
    #get posts from yrdsb
    profile = instaloader.Profile.from_username(bot.context, 'yrdsb.schools')
    posts = profile.get_posts()

    #check most recent post
    for i, post in enumerate(posts):
        if i > 0:
            break

        if post.mediaid != last_id: #check id
            return post
        
    return False




def check_posts():
    '''
        Indefinitely running function to check if new posts have been made, classify them, and send announcements if posts are deemed as indicating a snow day
        
        Args:
            None
            
        Returns:
            None
    '''
    
    #parse config info
    config = configparser.ConfigParser()
    config.read('F:\ics4u\projects\capstone\config_info.ini')


    while True: #run indefinitely
        insta_bot = instaloader.Instaloader()
        insta_bot.login(config['instagram']['username'], config['instagram']['password'])

        last_id = ""
        post = new_post(insta_bot, last_id) #check if new post has been made


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


#run bot
run_bot()
