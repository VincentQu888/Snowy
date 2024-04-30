import instaloader
import discord
import configparser
import pandas as pd
import csv


#parse config info
config = configparser.ConfigParser()
config.read('F:\ics4u\projects\capstone\config_info.ini')


#log in to instagram
bot = instaloader.Instaloader()
bot.login(config['instagram']['username'], config['instagram']['password'])


#get posts from yrdsb
profile = instaloader.Profile.from_username(bot.context, 'yrdsb.schools')
posts = profile.get_posts()


#write to csv file
with open('yrdsb_instagram_posts.csv', 'w', encoding="utf-8") as file:
    writer = csv.writer(file)
    field = ["description", "category"]
    
    writer.writerow(field)
    for post in posts:
        desc_text = post.caption.replace(",", "")
        writer.writerow([desc_text, "0"])